# C刷题

## OJ输入

没有输入：不需要任何的输入

一组输入：每次执行编译完的程序只会有一组输入输出

多组输入：每次执行编译完的程序都会有多组输入输出

```c
char ch = 0;
int i = 0;

while(~(ch = getchar()))
//while((ch = getchar()) != EOF)
{
    putchar(ch);
    //...
}

while(~scanf("%d", &i))
{
    //...
}
```


BC35

多组输入用while 还有换行