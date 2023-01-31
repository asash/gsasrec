#include <stdio.h>
#include <stdlib.h>

#define BUFFERSIZE 10240

void print(int * buf, int pos) {
    int i ;
    for (i = 0; i < pos; ++i)
        putchar(buf[i]) ;
    putchar('\n') ;
}

int main(int argc, char* argv[])
{
    int c ;
    int buf[BUFFERSIZE] ;
    int pos = 0 ;
    while((c = getchar()) != EOF)
    {
        switch (c)
        {
        case '\b':
        {
            if (pos > 0)
                pos-- ;
            break ;
        }
        case '\n':
        {
            print(buf, pos);
            pos = 0 ;
            break ;
        }

        case '\r':
        {
            print(buf, pos);
            pos = 0 ;
            break ;
        }

        default:
        {
            buf[pos++] = c ;
            break ;
        }
        }
    }
    return 0 ;
} 
