// Copyright (c) 2009-2011, Tor M. Aamodt,  Ali Bakhoda, Ivan Sham,
// Wilson W.L. Fung
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "stack.h"

#include <assert.h>
#include <stdlib.h>

void push_stack(Stack *S, address_type val) {
  assert(S->top < S->max_size);
  S->v[S->top] = val;
  (S->top)++;
}

address_type pop_stack(Stack *S) {
  (S->top)--;
  return (S->v[S->top]);
}

address_type top_stack(Stack *S) {
  assert(S->top >= 1);
  return (S->v[S->top - 1]);
}

Stack *new_stack(int size) {
  Stack *S;
  S = (Stack *)malloc(sizeof(Stack));
  S->max_size = size;
  S->top = 0;
  S->v = (address_type *)calloc(size, sizeof(address_type));
  return S;
}

void free_stack(Stack *S) {
  free(S->v);
  free(S);
}

int size_stack(Stack *S) { return S->top; }

int full_stack(Stack *S) { return S->top >= S->max_size; }

int empty_stack(Stack *S) { return S->top == 0; }

int element_exist_stack(Stack *S, address_type value) {
  int i;
  for (i = 0; i < S->top; ++i) {
    if (value == S->v[i]) {
      return 1;
    }
  }
  return 0;
}

void reset_stack(Stack *S) { S->top = 0; }
