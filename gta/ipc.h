#pragma once
#include <string>
#include <span>
#include <cstring>
#include <iostream>
#include <climits>
#include "google/protobuf/message.h"
#include "launch_debugger.h"
#include <immintrin.h>

using std::string;
using std::wstring;
using std::array;
using std::span;
using std::byte;
using std::string_view;
using google::protobuf::Message;

// Wrapper around Windows' memory-mapping methods
// Exposes pointer to memory mapped region as `void* Bytes` (i.e. write/read to MemoryMap -> Bytes)
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

// Memory mapped vector of bytes
// The first 16 bytes of shared memory hold its capacity and length, which are now accessible to other programs
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
		if (Capacity() == 0) ChangeCapacity(1024);
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

// DEPRECATED: See Event class
// Synchronization bitflags
#define BEGIN_TRAINING 0
#define REQUEST_GAME_STATE 1
#define UNSTUCK 2
#define RAYCASTS 3
#define RESET 4

struct Flags: MemoryMap
{
	#define FLAGS_TAGNAME "Flags"
	#define N_FLAGS 5

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
		while (GetFlag(Flag) != Value) _mm_pause();
	}
};


// Win32 Events-based synchronization of above
struct Event
{
	HANDLE Handle;

	Event(string Name, bool ManualReset = false, bool InitialState = false) :
		Handle(CreateEventA(NULL, ManualReset, InitialState, Name.c_str()))
	{
	}

	Event() = default;

	bool Set()
	{
		return SetEvent(Handle);
	}

	bool Reset()
	{
		return ResetEvent(Handle);
	}

	// Blocking wait
	DWORD Wait(bool Alertable = true)
	{
		DWORD result;
		// Alertable WaitForSingleObjectEx doesn't block, while WaitForSingleObjectEx does
		// Since we want this method to block, we wait until we get a response in the Alertable case
		if (Alertable) while (true) if ((result = WaitForSingleObjectEx(Handle, INFINITE, true)) != WAIT_IO_COMPLETION) return result;
		// Otherwise run the blocking method 
		else return WaitForSingleObjectEx(Handle, INFINITE, false);
	}
};

// DEPRECATED
// TODO: add some sort of assertion that the deserialized typename is the actual type's name
// A memory region that contains a protobuf object. Inherits read/write lock behavior from Protobuf library
template<std::derived_from<Message> T>
struct StructuredMemory : Memory
{
	inline static const std::string PayloadTypeName = std::string(T::GetDescriptor()->name());

	StructuredMemory(string Tagname) : Memory(Tagname) 
	{};

	StructuredMemory() = default;

	// Supports assignment-is-writing-memory when the operand is a protobuf Message object
	void operator = (const T& Msg)
	{
		Payload P = {};
		P.set_typename_(PayloadTypeName);
		P.set_data(Msg.SerializeAsString());
		Memory::operator=(P.SerializeAsString());
	}

	// Supports casting-is-reading-memory when the cast type is a protobuf Message object
	operator T ()
	{
		Payload P = {};
		T Message = {};
		P.ParseFromString(static_cast<string_view>(*this));
		Message.ParseFromString(P.data());
		return Message;
	}
};

// DEPRECATED
// A memory region containing a protobuf object with a specific synchronized access pattern ('request-locked')
// See scripts/ipc.py
template<std::derived_from<Message> T>
struct RequestLockedMemory : StructuredMemory<T>
{

	Event ReadFlag;
	Event WriteFlag;

	// Initializes RequestLockedMemory object with read and write access flags 
	RequestLockedMemory(string Tagname) : 
		StructuredMemory<T>(Tagname), 
		ReadFlag(Tagname + "Read"),
		WriteFlag(Tagname + "Write")
	{}

	// Supports assignment-is-writing-memory when the operand is a protobuf Message object
	void operator = (const T& Msg)
	{
		ReadFlag.Wait();
		StructuredMemory<T>::operator=(Msg);
		WriteFlag.Set();
	}

	// Supports casting-is-reading-memory when the cast type is a protobuf Message object
	operator T ()
	{
		ReadFlag.Set();
		WriteFlag.Wait();
		return static_cast<StructuredMemory<T>>(*this);
	}
};

