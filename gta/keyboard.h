#pragma once
#include "framework.h"
#include "ipc.h"

HWND GetCurrentWindow()
{
	return SwapChainDesc.OutputWindow;
}

enum KeyEvent 
{
	NONE, UP, DOWN
};

struct KeyboardEvent
{
	KeyEvent W, A, S, D;
	static KeyEvent Compare(bool Previous, bool Current)
	{
		if (!Previous && Current) return UP;
		if (Previous && !Current) return DOWN;
		return NONE;
	}

	KeyboardEvent(KeyboardState& Previous, KeyboardState& Current) :
		W(Compare(Previous.w(), Current.w())),
		A(Compare(Previous.a(), Current.a())),
		S(Compare(Previous.s(), Current.s())),
		D(Compare(Previous.d(), Current.d()))
	{}
};


// inputs only count if they xor each other
// model gets 8 keys: 4 wasd up, 4 wasd down. it picks one
// if the chosen value for the bit is the same as last time it's a no-op
static void SendKeypress()
{
	constexpr auto W = 0x57;
	constexpr auto A = 0x41;
	constexpr auto S = 0x53;
	constexpr auto D = 0x44;

	const HWND Window = GetCurrentWindow();
	static StructuredMemory<KeyboardState> K("Keyboard");
	auto KS = static_cast<KeyboardState>(K);
	
	#define PARSE_KEY(KEY, CODE) \
	if(KS.has_##KEY()) {\
		if (KS.##KEY()) PostMessage(Window, WM_KEYUP, CODE, 0xC0000001 & MapVirtualKey(CODE, 0));\
		if (!KS.##KEY()) PostMessage(Window, WM_KEYDOWN, CODE, 0);\
	}

	PARSE_KEY(w, 0x57);
	PARSE_KEY(a, 0x41);
	PARSE_KEY(s, 0x53);
	PARSE_KEY(d, 0x44);

}