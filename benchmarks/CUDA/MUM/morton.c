#include <stdio.h>

void morton(int i, int *x, int *y)
{
  *x = 0;
  *y = 0;

  int b;
  for (b = 0; b < 16; b++)
  {
    *x |= (i & (1 << (b*2)))     >> b;
    *y |= (i & (1 << (b*2+1))) >> (b+1);
  }
}

int main(int argc, char ** argv)
{
  int i;
  for (i = 0; i < 100; i++)
  {
    int x;
    int y;

    morton(i,&x,&y);

    printf("%d: %d %d\n", i, x, y);
  }

  return 0;
}
