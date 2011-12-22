#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// 641986 gets you ~30 MB of data
int main(int argc, char **argv)
{
  FILE *fp;
  int i=0, canes=0,num_files=0,j=0;
  int year,month,date,hour,num,speed,press;
  float lat,lon;
  int hours[4] = {0,6,12,18};
  char *name, fname[16];
  char names[21][10] = {"ALBERTO", "BERYL", "CHRIS","DEBBY","ERNESTO","FLORENCE","GORDON",
      "HELENE","ISAAC","JOYCE","KIRK","LESLIE","MICHAEL","NADINE","OSCAR","PATTY","RAFAEL",
      "SANDY","TONY","VALERIE","WILLIAM"};

  if(argc < 3)
  {
    fprintf(stderr,"we need a number of hurricanes to create and number of files to create\n");
    exit(0);
  }

  canes = atoi(argv[1]);
  num_files = atoi(argv[2]);

  canes = (canes / num_files) + 1;
  
  for(j=0;j<num_files;j++)
  {
    sprintf(fname, "data/cane%d_%d.db", num_files,j);
    
    if ((fp = fopen(fname, "w")) == NULL) {
		 fprintf(stderr, "Failed to open output file '%s'!\n", fname);
		 return -1;
	 }
  
    srand(time(NULL));

    for(i=0;i<canes;i++)
    {
      year = 1950 + rand() % 55;
      month = 1 + rand() % 12;
      date = 1 + rand() % 28;
      hour = hours[rand()%4];
      num = 1 + rand() % 28;
      name = names[rand()%21];
      lat = ((float)(7 + rand() % 63)) + ((float) rand() / (float) 0x7fffffff);
      lon = ((float)(rand() % 358)) + ((float) rand() / (float) 0x7fffffff); 
      speed = 10+ rand() % 155;
      press = rand() % 900;

      fprintf(fp, "%4d %2d %2d %2d %2d %-9s %5.1f %5.1f %4d %4d\n", year, month, date, hour, num, name, lat, lon, speed, press); 

    }

    fclose(fp);
  }

    return 0;

  }
