#pragma once
#include <string>
#include <span>
#include <cstring>
#include <iostream>
#include <climits>
//#include "cpp.pb.h"
#include "google/protobuf/message.h"
#include "launch_debugger.h"

using std::string;
using std::wstring;
using std::array;
using std::span;
using std::byte;
using std::string_view;
using google::protobuf::Message;

struct MemoryMap
{
	string Tagname;
	HANDLE Handle;
	void* Bytes;
	size_t Size;

	static HANDLE CreateHandle(string Tagname, size_t N)
	{
		return CreateFileMapping(
			INVALID_HANDLE_VALUE,
			NULL,
			PAGE_READWRITE,
			0,
			N,
			wstring(Tagname.begin(), Tagname.end()).c_str()
		);
	}

	static void* CreateMap(HANDLE Handle)
	{
		return MapViewOfFile(
			Handle,
			FILE_MAP_ALL_ACCESS,
			0,
			0,
			0
		);
	}

	MemoryMap(string Tagname, size_t N) :
		Size(N),
		Tagname(Tagname),
		Handle(CreateHandle(Tagname, N)),
		Bytes(CreateMap(Handle))
	{}

	void Resize(size_t N)
	{
		CloseHandle(Handle);
		Handle = CreateHandle(Tagname, N);
		auto Remapped = CreateMap(Handle);
		memcpy(Remapped, Bytes, Size);
		UnmapViewOfFile(Bytes);
		Bytes = Remapped;
		Size = N;
	}

	void Close()
	{
		CloseHandle(Handle);
		UnmapViewOfFile(Bytes);
	}

	void Flush()
	{
		FlushFileBuffers(Handle);
	}

};

struct Memory: MemoryMap
{
	static const size_t HEAD_LENGTH = 2 * sizeof(size_t);

	static size_t Capacity(string Tagname)
	{
		auto Temp = MemoryMap(Tagname, sizeof(size_t));
		size_t Capacity = ((size_t*)Temp.Bytes)[0];
		Temp.Close();
		return Capacity;
	}

	Memory(string Tagname) : MemoryMap(Tagname, max(HEAD_LENGTH, Capacity(Tagname)))
	{
		if (Capacity() == 0) ChangeCapacity(1);
	}

	inline size_t& Capacity() const
	{
		return ((size_t*)MemoryMap::Bytes)[0];
	}

	inline size_t& Length() const
	{
		return ((size_t*)MemoryMap::Bytes)[1];
	}

	inline void* Raw() const
	{
		return ((byte*)MemoryMap::Bytes) + HEAD_LENGTH;
	}

	void ChangeCapacity(size_t N)
	{
		MemoryMap::Resize(N + HEAD_LENGTH);
		Capacity() = N;
	}

	void operator=(const string& Other)
	{
		if (Other.size() > Capacity()) ChangeCapacity(Other.size());
		Length() = Other.size();
		memcpy(Raw(), Other.c_str(), Other.size());
	}

	operator string_view() const
	{
		return { (char*)Raw(), Length() };
	}
};

struct Flags: MemoryMap
{
	#define FLAGS_TAGNAME "Flags"
	#define N_FLAGS 2

	Flags() : MemoryMap(FLAGS_TAGNAME, (N_FLAGS + CHAR_BIT - 1) / CHAR_BIT) {};

	void SetFlag(int Flag, bool Value)
	{
		int Position = Flag / CHAR_BIT;
		int Offset = Flag % CHAR_BIT;
		byte Mask = ~(byte(1) << Offset);
		byte* Bytes = (byte*)MemoryMap::Bytes;
		Bytes[Position] = (Bytes[Position] & Mask) | ( (byte) Value << Offset);
		MemoryMap::Flush();
	}

	bool GetFlag(int Flag)
	{
		int Position = Flag / CHAR_BIT;
		int Offset = Flag % CHAR_BIT;
		byte Mask = byte(1) << Offset;
		byte State = ((byte*)MemoryMap::Bytes)[Position];
		return (State & Mask) != byte(0);
	}

	void WaitUntil(int Flag, bool Value)
	{
		while (GetFlag(Flag) != Value) Sleep(1);
	}
};

// TODO: add some sort of assertion that the deserialized typename is the actual type's name
template<std::derived_from<Message> T>
struct StructuredMemory : Memory
{
	inline static const std::string PayloadTypeName = std::string(T::GetDescriptor()->name());

	StructuredMemory(string Tagname) : Memory(Tagname) 
	{};

	StructuredMemory() = default;

	void operator = (const T& Msg)
	{
		Payload P = {};
		P.set_typename_(PayloadTypeName);
		P.set_data(Msg.SerializeAsString());
		Memory::operator=(P.SerializeAsString());
	}

	operator T ()
	{
		Payload P = {};
		T Message = {};
		P.ParseFromString(static_cast<string_view>(*this));
		Message.ParseFromString(P.data());
		return Message;
	}
};

#define BEGIN_TRAINING 0
#define REQUEST_GAME_STATE 1

template<std::derived_from<Message> T, size_t RequestFlag>
struct RequestLockedMemory : StructuredMemory<T>
{

	Flags flags;

	using StructuredMemory<T>::StructuredMemory;

	void operator = (const T& Msg)
	{
		flags.WaitUntil(RequestFlag, true);
		StructuredMemory<T>::operator=(Msg);
		flags.SetFlag(RequestFlag, false);
	}

	operator T ()
	{
		flags.SetFlag(RequestFlag, true);
		flags.WaitUntil(RequestFlag, false);
		return static_cast<StructuredMemory<T>>(*this);
	}
};

