#include <stdio.h>
#include <sys/stat.h>
#include <dirent.h>
#include <libgen.h>
extern "C" {
#include <gfal_api.h>
#include <lcg_util.h>
}
#include <string.h>

int recursiveDelete(char* dirname) {

  DIR *dp;
  struct dirent *ep;

  char abs_filename[FILENAME_MAX];

  dp = gfal_opendir(dirname);
  if (dp != NULL)
    {
      while (ep = gfal_readdir (dp)) {
        struct stat stFileInfo;

        snprintf(abs_filename, FILENAME_MAX, "%s/%s", dirname, ep->d_name);

        if (gfal_stat(abs_filename, &stFileInfo) < 0)
          perror( abs_filename );

        if(S_ISDIR(stFileInfo.st_mode)) {
          if(strcmp(ep->d_name, ".") && 
             strcmp(ep->d_name, "..")) {
            printf("%s directory\n",abs_filename);
            recursiveDelete(abs_filename);
          }
        } else {
	  int result = lcg_del(abs_filename, 5, NULL,NULL,NULL,0,0); 
          printf("%s\n",abs_filename);
	  if(result==-1)
	  { 
	    printf("Delete failed, invoking lcg_del %s file --> result %d\n",abs_filename, result); 
	    printf("------------------------------------------------------------------------------------------------------\n");

	  }  
        }
          }
      (void) gfal_closedir (dp);
        }
  else
    perror ("Couldn't open the directory");

  printf("%s\n",dirname);
  int result = gfal_rmdir(dirname);
  if(result==-1)
  { 
    printf("smiple delete directory failed, invoking lcg_del %s file --> result %d\n",abs_filename, result); 
    printf("------------------------------------------------------------------------------------------------------\n");
  }  

  return 0;

}

int main( int argc, char *argv[] )
{
    printf("Argumentos : %d\n", argc);
    if(argc==2)
      recursiveDelete(argv[1]);
  
    return 0;
}
