#pragma once

#ifdef FIR_DLL_EXPORTS
#define FIR_DLL_API __declspec(dllexport)
#else
#define FIR_DLL_API __declspec(dllimport)
#endif

//extern "C" MYDLL_API void FIR(const float x[], const float coeffTaps[], float y[]);
extern "C" FIR_DLL_API void FIR(const float x[],
    const float coeffTaps[],
    float y[],
    int x_size,
    int num_taps);