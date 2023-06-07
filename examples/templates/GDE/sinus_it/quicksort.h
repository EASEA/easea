//Quicksort algorithm to sort the sines of the individual according to growing frequencies
void quicksort_freq(double* tab_sines,int first,int last){
   int i, j, pivot;
   double temp_amp, temp_freq, temp_ph, temp_dec;

   if(first<last){
      pivot=first;
      i=first;
      j=last;

      while(i<j){
         while(tab_sines[4*i+1]>=tab_sines[4*pivot+1]&&i<last)
            i++;
         while(tab_sines[4*j+1]<tab_sines[4*pivot+1])
            j--;
         if(i<j){
            temp_amp=tab_sines[4*i];
            temp_freq=tab_sines[4*i+1];
            temp_ph=tab_sines[4*i+2];
            temp_dec=tab_sines[4*i+3];
            tab_sines[4*i]=tab_sines[4*j];
            tab_sines[4*i+1]=tab_sines[4*j+1];
            tab_sines[4*i+2]=tab_sines[4*j+2];
            tab_sines[4*i+3]=tab_sines[4*j+3];
            tab_sines[4*j]=temp_amp;
            tab_sines[4*j+1]=temp_freq;
            tab_sines[4*j+2]=temp_ph;
            tab_sines[4*j+3]=temp_dec;
         }
      }

      temp_amp=tab_sines[4*pivot];
      temp_freq=tab_sines[4*pivot+1];
      temp_ph=tab_sines[4*pivot+2];
      temp_dec=tab_sines[4*pivot+3];
      tab_sines[4*pivot]=tab_sines[4*j];
      tab_sines[4*pivot+1]=tab_sines[4*j+1];
      tab_sines[4*pivot+2]=tab_sines[4*j+2];
      tab_sines[4*pivot+3]=tab_sines[4*j+3];
      tab_sines[4*j]=temp_amp;
      tab_sines[4*j+1]=temp_freq;
      tab_sines[4*j+2]=temp_ph;
      tab_sines[4*j+3]=temp_dec;

      //Recursive calls
      quicksort_freq(tab_sines,first,j-1);
      quicksort_freq(tab_sines,j+1,last);

   }
}
