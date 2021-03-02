
	/* 
	** Generates simulated greyscale video data of a white filled
	** object moving randomly in four directions (up, down, left, right).
	** Motions are generally in the given direction but with some
	** amount of noise.
	** Motions occur at a steady velocity of 1 pixel per frame, for
	** a random number of 10-30 frames.  Direction changes are
	** constrained to remain within the image bounds.
	**
	** The purpose of this data is to debug/test a neural network
	** that learns class labels from video.  It should be able to
	** achieve near perfect accuracy.
	*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define	SQR(x)	((x)*(x))

	/* These first few functions are for random number generators. */

static unsigned int rand_x[56],rand_y[256],rand_z; /* used for normal dist */
static int rand_j, rand_k;		/* used for normal dist */


unsigned urand0 (void)
{
if (--rand_j == 0) rand_j = 55;
if (--rand_k == 0) rand_k = 55;
return rand_x[rand_k] += rand_x[rand_j];
}

void init_generators(unsigned seed)
{
int i;

		/* inits the normal generator */
rand_x[1] = 1;
if(seed)
  rand_x[2] = seed;
else
  rand_x[2] = time (NULL);
for (i=3; i<56; ++i) rand_x[i] = rand_x[i-1] + rand_x[i-2];
rand_j = 24;
rand_k = 55;
for (i=255; i>=0; --i)
  urand0 (); //run loop for a while
for (i=255; i>=0; --i)
  rand_y[i] = urand0 ();
rand_z = urand0 ();

		/* inits the uniform generator */
srand( (unsigned)time( NULL ) );
}

unsigned urand (void)
{
int i;

i = rand_z % 256;
rand_z = rand_y[i];
if (--rand_j == 0) rand_j = 55;
if (--rand_k == 0) rand_k = 55;
rand_y[i] = rand_x[rand_k] += rand_x[rand_j];
return rand_z;
}

		/* returns a value from a normal dist with mean=0 stddev=1 */

double normal_rand (void)
{
static int flag = 0;
static double z, a = 2147483648.0;
double v1, v2, s;

if (flag) {
  flag = 0;
  return z;
  }
flag = 1;
do {
  v1 = urand()/a - 1;
  v2 = urand()/a - 1;
  }
while ((s = v1*v1 + v2*v2) > 1.0);
s = sqrt (-2.0 * log(s) / s);
z = v1 * s;
return v2 * s;
}

		/* returns a value from -4...+4 */

double uniform_rand (void)
{
int		randgen;
double	ret;

randgen=rand();
ret=((double)randgen/(double)RAND_MAX*8.0)-4.0;
return ret;
}



#define	FRAMES	8*60*10		/* total frames produced in video */
#define	ROWS	128		/* size of image in video */
#define	COLS	128		/* size of image in video */
#define	RADIUS	5.0		/* size of moving object */
#define	DEBUG	0		/* 0=>print out info, 1=>save images */

int main(int argc, char *argv[])

{
double		x,y;	/* coordinates of moving object */
unsigned char	*image;
double		delta_x,delta_y;	/* vector of motion */
double		delta_t;		/* #frames to move in this vector */
int		moving_count;		/* time accumulated along vector */
int		change;			/* boolean flag to switch direction */
int		state;			/* 0=>up, 1=>right, 2=>down, 3=>left */
int		state_ok;		/* check to stay in bounds */
double		x2,y2;
int		i,j;
double		dist;
FILE		*fpt, *fpt_gt;
char		filename[320];		/* ppm image filename */
char		jfilename[320];		/* jpeg image filename */
char		gt_filename[320];		/* ground truth filename */
char		command[640];		/* command to convert ppm to jpeg */
char    folder[320] = "data/test";
int     speedup = 3;
init_generators(0);

image=(unsigned char *)calloc(ROWS*COLS,1);

	/* initialize object to center of image */
x=(double)(COLS)/2.0;
y=(double)(ROWS)/2.0;
change=1;	/* binary flag 1=>change direction, 0=>don't change */

sprintf(gt_filename,"%s/gt_frame.txt",folder);
fpt_gt = fopen(gt_filename,"w");
for (j=0; j<FRAMES; j++)
  {
	/* create image for this frame */
  for (i=0; i<ROWS*COLS; i++)
    image[i]=0;		/* background is black */
  for (y2=y-RADIUS; y2<=y+RADIUS; y2+=1.0)
    {
    for (x2=x-RADIUS; x2<=x+RADIUS; x2+=1.0)
      {
      dist=sqrt(SQR(x-x2)+SQR(y-y2));
      if (dist > RADIUS)	/* object will be circle */
        continue;	/* comment this out to change object to square */
      image[(int)y2*COLS+(int)x2]=255;	/* foreground is white */
      }
    }
	/* write out frame */
  if (DEBUG)	/* switch this flag to debug, turning off saving files */
    {	/* print out positions for debugging */
    printf("%lf\t%lf\t%d\n",x,y,state);
    }
  else
    {	/* save out images */
    sprintf(filename,"%s/frame%05d.ppm",folder, j);
    fpt=fopen(filename,"wb");
    fprintf(fpt,"P5 %d %d 255\n",COLS,ROWS);
    fwrite(image,ROWS*COLS,1,fpt);
    fclose(fpt);
    //sprintf(jfilename,"frame%04d.jpg",j);
    //sprintf(command,"cjpeg %s > %s",filename,jfilename);
    //system(command);
    fprintf(fpt_gt,"frame%05d.ppm\t%d\t%lf\t%lf\n",j,state,x,y);
    }
    
	/* if time to change direction, calculate new state and vector */
  if (change == 1)
    {

   	/* find random amount of time to move this direction */
    delta_t=normal_rand()*2.5+5.0;;
    if (delta_t < 3.0)
      delta_t=3.0;	/* min time to stay this direction */
    if (delta_t > 10.0)
      delta_t=10.0;	/* max time to stay this direction */

    state_ok=0;			/* check if random direction will
				            ** stay within image bounds */
    while (state_ok == 0)	/* keep trying new state until it's okay */
      {
      state=(int)fabs(uniform_rand());	/* 0=>up, 1=>right, 2=>down, 3=>left */
      if (state > 3)
	      state=3;		/* should be 0...3 after this */
      state_ok=1;		/* assume ok, now check for problems */
      if (state == 0  &&  y < speedup*1.0*delta_t+RADIUS)
        state_ok=0;		/* try another direction */
      if (state == 1  &&  x > (double)COLS-(speedup*1.0*delta_t+RADIUS))
        state_ok=0;		/* try another direction */
      if (state == 2  &&  y > (double)ROWS-(speedup*1.0*delta_t+RADIUS))
        state_ok=0;		/* try another direction */
      if (state == 3  &&  x < speedup*1.0*delta_t+RADIUS)
        state_ok=0;		/* try another direction */
      }
    moving_count=0;	/* cumulative time spent moving this direction */
	  /* find new vector of motion */
    if (state == 0)
      {
      delta_x=normal_rand()/3.0;
      delta_y=-speedup*1.0;
      }
    if (state == 1)
      {
      delta_x=speedup*1.0;
      delta_y=normal_rand()/3.0;
      }
    if (state == 2)
      {
      delta_x=normal_rand()/3.0;
      delta_y=speedup*1.0;
      }
    if (state == 3)
      {
      delta_x=-speedup*1.0;
      delta_y=normal_rand()/3.0;
      }
    if (DEBUG)
      printf("new state %d vector %.2lf %.2lf duration %.0lf\n",
		state,delta_x,delta_y,delta_t);
    change=0;	/* reset change flag */
    }
    
	/* move object in given direction, update time spent moving */
  x+=delta_x;
  y+=delta_y;
  moving_count++;
  if (moving_count >= (int)delta_t)
    change=1;	/* time to change directions */
  }
fclose(fpt_gt);
}
