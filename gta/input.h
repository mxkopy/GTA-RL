#pragma once
#include "framework.h"
#include "ipc.h"

HWND GetCurrentWindow()
{
	return SwapChainDesc.OutputWindow;
}

struct Key
{
	// true -> key is down; false -> key is up
	bool State;

	const unsigned long long Code;
	const unsigned long long ScanCode;

	Key(unsigned int Code) : State(false), Code(Code), ScanCode(MapVirtualKey(Code, MAPVK_VK_TO_VSC)) {};
	
	void SendUp()
	{
		unsigned long long Modifier = 0xC000;
		unsigned long long LP = (Modifier | ScanCode) << 16;
		PostMessage(GetCurrentWindow(), WM_KEYUP, Code, 0x0001 | LP);
		State = false;
	}

	void SendDown()
	{
		unsigned long long Modifier = State ? 0x4000 : 0x0000;
		unsigned long long LP = (Modifier | ScanCode) << 16;
		PostMessage(GetCurrentWindow(), WM_KEYDOWN, Code, 0x0001 | LP);
		//PostMessage(GetCurrentWindow(), WM_CHAR, NULL, 0x0001 | LP);
		State = true;
	}

	bool Update(bool KeyPress)
	{
		if (KeyPress) SendDown();
		if (State && !KeyPress) SendUp();
		return State;
	}
};

static void SendKeypress()
{

	static Key w(0x57);
	static Key a(0x41);
	static Key s(0x53);
	static Key d(0x44);

	static StructuredMemory<KeyboardState> K("Keyboard");

	auto KS = static_cast<KeyboardState>(K);

	#define PARSE_KEY(KEY) KEY.Update(KS.##KEY());
	
	PARSE_KEY(w);
	PARSE_KEY(a);
	PARSE_KEY(s);
	PARSE_KEY(d);

}