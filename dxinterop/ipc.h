#pragma once
#include "framework.h"
#include "google/protobuf/message.h"

using std::string;
using std::array;
using std::span;
using std::byte;
using google::protobuf::Message;


struct MemoryMap
{
	HANDLE Handle;
	void* Bytes;
	string Tagname;

	MemoryMap(string Tagname, size_t N)
	{
		auto WTagname = std::wstring(Tagname.begin(), Tagname.end());
		HANDLE Handle = CreateFileMapping(
			INVALID_HANDLE_VALUE,
			NULL,
			PAGE_READWRITE,
			0,
			N,
			WTagname.c_str()
		);
		Bytes = MapViewOfFile(Handle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
	}

	void Close()
	{
		CloseHandle(Handle);
		UnmapViewOfFile(Bytes);
	}

};

struct Memory 
{

	MemoryMap M;

	static size_t Capacity(string Tagname)
	{
		auto Temp = MemoryMap(Tagname, sizeof(size_t));
		size_t Capacity = ((size_t*)Temp.Bytes)[0];
		Temp.Close();
		return Capacity;
	}

	Memory(string Tagname) : M(MemoryMap(Tagname, max(2*sizeof(size_t), Capacity(Tagname))))
	{
		if (Capacity() == 0) ChangeCapacity(1);
	}

	size_t& Capacity()
	{
		return ((size_t*)M.Bytes)[0];
	}

	size_t& Length()
	{
		return ((size_t*)M.Bytes)[1];
	}

	void* Raw()
	{
		return (byte*)M.Bytes + 2 * sizeof(size_t);
	}

	void ChangeCapacity(size_t N)
	{
		CloseHandle(M.Handle);
		auto Resized = MemoryMap(M.Tagname, N + 2*sizeof(size_t));
		memcpy(Resized.Bytes, M.Bytes, min(Capacity(), N));
		UnmapViewOfFile(M.Bytes);
		M.Handle = Resized.Handle;
		M.Bytes = Resized.Bytes;
		Capacity() = N;
	}

	struct {

		template<size_t N>
		span<byte, N> & operator = (const span<byte, N> Other)
		{
			while (N >= Capacity()) ChangeCapacity(Capacity() * 2);
			byte* Bytes = memcpy(Raw(), Other.begin(), N);
			Length() = N;
			return span<byte, N>(Bytes, Bytes + N);
		}

		operator span<byte> () const 
		{
			MemoryMap& M = M;
			byte* Bytes = (byte*) M.Bytes;
			size_t N = ((size_t*)M.Bytes)[1];
			return span<byte>(
				Bytes + 2 * sizeof(size_t),
				Bytes + 2 * sizeof(size_t) + N
			);
		}
	} Bytes;

};

struct StructuredMemory : Memory
{
	Payload P = {};

	template<std::derived_from<Message> T>
	T& operator = (T& Msg)
	{
		P.set_data(Msg.SerializeAsString())
		P.SerializeToArray(Raw(), Msg.ByteSize);
		return Msg;
	}

	template<std::derived_from<Message> T>
	operator T () const
	{
		return Msg.ParseFromArray(Raw(), Length());
	}

};