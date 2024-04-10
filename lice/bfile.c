/** standalone bfile.c
 *
 * MicroHs's bfile.c is not written to be compiled as a standalone object (it
 * is directly #included in eval.c), so this file adapts it to be.
 *
 * First, it includes some standard C headers that MicroHs depends on.
 * Then, it defines some configuration variables, and implements some hacks.
 * Then, it replicates some the definitions from eval.c that bfile.c needs.
 * Finally, it contains the verbatim contents of bfile.c.
 */

/***************** ported from eval.c *******************/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/***************** CONFIGS AND HACKS *******************/
// We want want stdio
#define WANT_STDIO 1

// We don't want to inline anything, because we need to link
#define INLINE

// bfile.c defines some important functions as static. No!
#define static

// Getters for standard FILE pointers, for
FILE *mystdin() { return stdin; }
FILE *mystdout() { return stdout; }
FILE *mystderr() { return stderr; }


/***************** ported from eval.c *******************/
typedef intptr_t value_t;       /* Make value the same size as pointers, since they are in a union */

#if !defined(INLINE)
#define INLINE inline
#endif  /* !define(INLINE) */

#if !defined(NORETURN)
#define NORETURN __attribute__ ((noreturn))
#endif /* !defined(NORETURN) */

#if !defined(MALLOC)
#define MALLOC malloc
#endif

#if !defined(FREE)
#define FREE free
#endif

#if !defined(EXIT)
#define EXIT exit
#endif

#if !defined(ERR)
#define ERR(s)    do { fprintf(stderr,"ERR: "s"\n");   EXIT(1); } while(0)
#define ERR1(s,a) do { fprintf(stderr,"ERR: "s"\n",a); EXIT(1); } while(0)
#endif  /* !define(ERR) */

NORETURN
void
memerr(void)
{
  ERR("Out of memory");
  EXIT(1);
}

/***************** BFILE *******************/
/*
 * BFILE is used to access files.
 * It allows various "transducers" to be added
 * to the processing.
 */

/* Sanity checking */
#define CHECKBFILE(p, f) do { if (p->getb != f) ERR("CHECKBFILE"); } while(0)

/* BFILE will have different implementations, they all have these methods */
typedef struct BFILE {
  int (*getb)(struct BFILE*);
  void (*ungetb)(int, struct BFILE*);
  void (*putb)(int, struct BFILE*);
  void (*flushb)(struct BFILE*);
  void (*closeb)(struct BFILE*);
} BFILE;

INLINE int
getb(BFILE *p)
{
  return p->getb(p);
}

static INLINE void
ungetb(int c, BFILE *p)
{
  p->ungetb(c, p);
}

static INLINE void
putb(int c, BFILE *p)
{
  p->putb(c, p);
}

static INLINE void
closeb(BFILE *p)
{
  p->closeb(p);
}

static INLINE void
flushb(BFILE *p)
{
  p->flushb(p);
}

void
putsb(const char *str, struct BFILE *p)
{
  char c;
  while ((c = *str++))
    putb(c, p);
}

size_t
readb(void *abuf, size_t size, BFILE *p)
{
  uint8_t *buf = abuf;
  size_t s;
  for (s = 0; s < size; s++) {
    int c = getb(p);
    if (c < 0)
      break;
    buf[s] = c;
  }
  return s;
}

/* convert -n to a string, handles MINBOUND correctly */
void
putnegb(value_t n, BFILE *p)
{
  int c = '0' - n % 10;
  if (n <= -10) {
    putnegb(n / 10, p);
  }
  putb(c, p);
}

void
putdecb(value_t n, BFILE *p)
{
  if (n < 0) {
    putb('-', p);
    putnegb(n, p);
  } else {
    putnegb(-n, p);
  }
}

/***************** BFILE from static buffer *******************/
struct BFILE_buffer {
  BFILE    mets;
  size_t   b_size;
  size_t   b_pos;
  uint8_t  *b_buffer;
};

int
getb_buf(BFILE *bp)
{
  struct BFILE_buffer *p = (struct BFILE_buffer *)bp;
  CHECKBFILE(bp, getb_buf);
  if (p->b_pos >= p->b_size)
    return -1;
  return p->b_buffer[p->b_pos++];
}

void
ungetb_buf(int c, BFILE *bp)
{
  struct BFILE_buffer *p = (struct BFILE_buffer *)bp;
  CHECKBFILE(bp, getb_buf);
  if (p->b_pos == 0)
    ERR("ungetb");
  p->b_buffer[--p->b_pos] = (uint8_t)c;
}

void
closeb_buf(BFILE *bp)
{
  CHECKBFILE(bp, getb_buf);
  FREE(bp);
}

/* There is no open().  Only used with statically allocated buffers. */
struct BFILE*
openb_buf(uint8_t *buf, size_t len)
{
  struct BFILE_buffer *p = MALLOC(sizeof(struct BFILE_buffer));;
  if (!p)
    memerr();
  p->mets.getb = getb_buf;
  p->mets.ungetb = ungetb_buf;
  p->mets.putb = 0;
  p->mets.flushb = 0;
  p->mets.closeb = closeb_buf;
  p->b_size = len;
  p->b_pos = 0;
  p->b_buffer = buf;
  return (struct BFILE*)p;
}

#if WANT_STDIO
/***************** BFILE via FILE *******************/
struct BFILE_file {
  BFILE    mets;
  FILE    *file;
};

int
getb_file(BFILE *bp)
{
  struct BFILE_file *p = (struct BFILE_file *)bp;
  CHECKBFILE(bp, getb_file);
  return fgetc(p->file);
}

void
ungetb_file(int c, BFILE *bp)
{
  struct BFILE_file *p = (struct BFILE_file *)bp;
  CHECKBFILE(bp, getb_file);
  ungetc(c, p->file);
}

void
putb_file(int c, BFILE *bp)
{
  struct BFILE_file *p = (struct BFILE_file *)bp;
  CHECKBFILE(bp, getb_file);
  (void)fputc(c, p->file);
}

void
flushb_file(BFILE *bp)
{
  struct BFILE_file *p = (struct BFILE_file *)bp;
  CHECKBFILE(bp, getb_file);
  fflush(p->file);
}

void
closeb_file(BFILE *bp)
{
  struct BFILE_file *p = (struct BFILE_file *)bp;
  CHECKBFILE(bp, getb_file);
  fclose(p->file);
  FREE(p);
}

void
freeb_file(BFILE *bp)
{
  struct BFILE_file *p = (struct BFILE_file *)bp;
  CHECKBFILE(bp, getb_file);
  FREE(p);
}

BFILE *
add_FILE(FILE *f)
{
  struct BFILE_file *p = MALLOC(sizeof (struct BFILE_file));
  if (!p)
    memerr();
  p->mets.getb   = getb_file;
  p->mets.ungetb = ungetb_file;
  p->mets.putb   = putb_file;
  p->mets.flushb = flushb_file;
  p->mets.closeb = closeb_file;
  p->file = f;
  return (BFILE*)p;
}
#endif


#if WANT_LZ77
/***************** BFILE via simple LZ77 decompression *******************/
struct BFILE_lz77 {
  BFILE    mets;
  BFILE    *bfile;              /* underlying BFILE */
  uint8_t  *buf;
  size_t   len;
  size_t   pos;
  int      read;
};

int
getb_lz77(BFILE *bp)
{
  struct BFILE_lz77 *p = (struct BFILE_lz77*)bp;
  CHECKBFILE(bp, getb_lz77);
  if (p->pos >= p->len)
    return -1;
  return p->buf[p->pos++];
}

void
ungetb_lz77(int c, BFILE *bp)
{
  struct BFILE_lz77 *p = (struct BFILE_lz77*)bp;
  CHECKBFILE(bp, getb_lz77);
  p->pos--;
}

void
putb_lz77(int b, BFILE *bp)
{
  struct BFILE_lz77 *p = (struct BFILE_lz77*)bp;
  CHECKBFILE(bp, getb_lz77);

  if (p->pos >= p->len) {
    p->len *= 2;
    p->buf = realloc(p->buf, p->len);
    if (!p->buf)
      memerr();
  }
  p->buf[p->pos++] = b;
}

void
closeb_lz77(BFILE *bp)
{
  struct BFILE_lz77 *p = (struct BFILE_lz77*)bp;
  CHECKBFILE(bp, getb_lz77);

  if (!p->read) {
    /* We are in write mode, so compress and push it down */
    uint8_t *obuf;
    size_t olen = lz77c(p->buf, p->pos, &obuf);
    FREE(p->buf);
    for (size_t i = 0; i < olen; i++) {
      putb(obuf[i], p->bfile);
    }
    FREE(obuf);
  }

  closeb(p->bfile);
  FREE(p);
}

void
flushb_lz77(BFILE *bp)
{
  /* There is nothing we can do */
}

BFILE *
add_lz77_decompressor(BFILE *file)
{
  struct BFILE_lz77 *p = MALLOC(sizeof(struct BFILE_lz77));

  if (!p)
    memerr();
  memset(p, 0, sizeof(struct BFILE_lz77));
  p->mets.getb = getb_lz77;
  p->mets.ungetb = ungetb_lz77;
  p->mets.putb = 0;
  p->mets.flushb = 0;
  p->mets.closeb = closeb_lz77;
  p->read = 1;
  p->bfile = file;

  size_t size = 25000;
  uint8_t *buf = MALLOC(size);
  size_t i;
  if (!buf)
    memerr();
  for(i = 0;;) {
    int b = getb(file);
    if (b < 0)
      break;
    if (i >= size) {
      size *= 2;
      buf = realloc(buf, size);
      if (!buf)
        memerr();
    }
    buf[i++] = b;
  }
  p->len = lz77d(buf, i, &p->buf);
  FREE(buf);
  p->pos = 0;
  return (BFILE*)p;
}

BFILE *
add_lz77_compressor(BFILE *file)
{
  struct BFILE_lz77 *p = MALLOC(sizeof(struct BFILE_lz77));

  if (!p)
    memerr();
  memset(p, 0, sizeof(struct BFILE_lz77));
  p->mets.getb = getb_lz77;
  p->mets.ungetb = 0;
  p->mets.putb = putb_lz77;
  p->mets.flushb = flushb_lz77;
  p->mets.closeb = closeb_lz77;
  p->read = 0;
  p->bfile = file;

  p->len = 25000;
  p->buf = MALLOC(p->len);
  if (!p->buf)
    memerr();
  p->pos = 0;
  return (BFILE*)p;
}

#endif  /* WANT_LZ77 */
/***************** BFILE with UTF8 encode/decode *******************/

struct BFILE_utf8 {
  BFILE    mets;
  BFILE    *bfile;
  int      unget;
};

/* This is not right with WORD_SIZE==16 */
int
getb_utf8(BFILE *bp)
{
  struct BFILE_utf8 *p = (struct BFILE_utf8*)bp;
  CHECKBFILE(bp, getb_utf8);
  int c1, c2, c3, c4;

  /* Do we have an ungetb character? */
  if (p->unget >= 0) {
    c1 = p->unget;
    p->unget = -1;
    return c1;
  }
  c1 = getb(p->bfile);
  if (c1 < 0)
    return -1;
  if ((c1 & 0x80) == 0)
    return c1;
  c2 = getb(p->bfile);
  if (c2 < 0)
    return -1;
  if ((c1 & 0xe0) == 0xc0)
    return ((c1 & 0x1f) << 6) | (c2 & 0x3f);
  c3 = getb(p->bfile);
  if (c3 < 0)
    return -1;
  if ((c1 & 0xf0) == 0xe0)
    return ((c1 & 0x0f) << 12) | ((c2 & 0x3f) << 6) | (c3 & 0x3f);
  c4 = getb(p->bfile);
  if (c4 < 0)
    return -1;
  if ((c1 & 0xf8) == 0xf0)
    return ((c1 & 0x07) << 18) | ((c2 & 0x3f) << 12) | ((c3 & 0x3f) << 6) | (c4 & 0x3f);
  ERR("getb_utf8");
}

void
ungetb_utf8(int c, BFILE *bp)
{
  struct BFILE_utf8 *p = (struct BFILE_utf8*)bp;
  CHECKBFILE(bp, getb_utf8);
  if (p->unget >= 0)
    ERR("ungetb_utf8");
  p->unget = c;
}

void
putb_utf8(int c, BFILE *bp)
{
  struct BFILE_utf8 *p = (struct BFILE_utf8 *)bp;
  CHECKBFILE(bp, getb_utf8);
  if (c < 0)
    ERR("putb_utf8: < 0");
  if (c < 0x80) {
    putb(c, p->bfile);
    return;
  }
  if (c < 0x800) {
    putb(((c >> 6 )       ) | 0xc0, p->bfile);
    putb(((c      ) & 0x3f) | 0x80, p->bfile);
    return;
  }
  if (c < 0x10000) {
    putb(((c >> 12)       ) | 0xe0, p->bfile);
    putb(((c >> 6 ) & 0x3f) | 0x80, p->bfile);
    putb(((c      ) & 0x3f) | 0x80, p->bfile);
    return;
  }
  if (c < 0x110000) {
    putb(((c >> 18)       ) | 0xf0, p->bfile);
    putb(((c >> 12) & 0x3f) | 0x80, p->bfile);
    putb(((c >> 6 ) & 0x3f) | 0x80, p->bfile);
    putb(((c      ) & 0x3f) | 0x80, p->bfile);
    return;
  }
  ERR("putb_utf8");
}

void
flushb_utf8(BFILE *bp)
{
  struct BFILE_utf8 *p = (struct BFILE_utf8*)bp;
  CHECKBFILE(bp, getb_utf8);

  flushb(p->bfile);
}

void
closeb_utf8(BFILE *bp)
{
  struct BFILE_utf8 *p = (struct BFILE_utf8*)bp;
  CHECKBFILE(bp, getb_utf8);

  closeb(p->bfile);
  FREE(p);
}

BFILE *
add_utf8(BFILE *file)
{
  struct BFILE_utf8 *p = MALLOC(sizeof(struct BFILE_utf8));

  if (!p)
    memerr();
  p->mets.getb = getb_utf8;
  p->mets.ungetb = ungetb_utf8;
  p->mets.putb = putb_utf8;
  p->mets.flushb = flushb_utf8;
  p->mets.closeb = closeb_utf8;
  p->bfile = file;
  p->unget = -1;

  return (BFILE*)p;
}
