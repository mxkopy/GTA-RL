#pragma once
#include <windows.h>
#include <cstring>
#include <string_view>
#include "cpp.pb.h"
#include "google/protobuf/message.h"

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

struct Memory 
{
	static const size_t HEAD_LENGTH = 2 * sizeof(size_t);

	MemoryMap M;

	static size_t Capacity(string Tagname)
	{
		auto Temp = MemoryMap(Tagname, sizeof(size_t));
		size_t Capacity = ((size_t*)Temp.Bytes)[0];
		Temp.Close();
		return Capacity;
	}

	size_t& Capacity() const
	{
		return ((size_t*)M.Bytes)[0];
	}

	size_t& Length() const
	{
		return ((size_t*)M.Bytes)[1];
	}

	void* RawBytes() const
	{
		return ((byte*)M.Bytes) + HEAD_LENGTH;
	}

	void ChangeCapacity(size_t N)
	{
		M.Resize(N + HEAD_LENGTH);
		Capacity() = N;
	}

	void Flush()
	{
		M.Flush();
	}

	struct {

		Memory& M;

		void operator=(const string& Other)
		{
			while (Other.size() >= M.Capacity()) M.ChangeCapacity(M.Capacity() * 2);
			M.Length() = Other.size();
			memmove(M.RawBytes(), Other.c_str(), Other.size());
		}

		operator string_view() const
		{
			return { (char*)M.RawBytes(), M.Length() };
		}

	} Bytes = { *this };

	Memory(string Tagname) : M(MemoryMap(Tagname, max(2 * sizeof(size_t), Capacity(Tagname))))
	{
		if (Capacity() == 0) ChangeCapacity(1);
	}

};

// TODO: add some sort of assertion that the deserialized typename is the actual type's name
template<std::derived_from<Message> T>
struct StructuredMemory : Memory
{
	inline static const std::string PayloadTypeName = std::string(T::GetDescriptor()->name());

	Payload P = {};

	StructuredMemory() = default;

	StructuredMemory(string Tagname) : Memory(Tagname) 
	{
		T Data{};
		P.set_typename_(PayloadTypeName);
		P.set_data(Data.SerializeAsString());
		Memory::Bytes = P.SerializeAsString();
	};

	void operator = (T& Msg)
	{
		P.set_data(Msg.SerializeAsString());
		Memory::Bytes = P.SerializeAsString();
	}

	operator T ()
	{
		T Message = {};
		P.ParseFromString(Memory::Bytes);
		Message.ParseFromString(P.data());
		return Message;
	}
};
