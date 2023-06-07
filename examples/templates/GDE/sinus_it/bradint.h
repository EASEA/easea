// Data structure in order to access the internal representation of the double number
typedef union {
    double d;
    unsigned long int uln_d;
} UDoubleBit;

//64-bits mask for the host (65 elements from 0 to 64 included)
unsigned long long int ulln_hmasks64[65] = {
  0x0000000000000000,
  0x0000000000000001,
  0x0000000000000003,
  0x0000000000000007,
  0x000000000000000F,
  0x000000000000001F,
  0x000000000000003F,
  0x000000000000007F,
  0x00000000000000FF,
  0x00000000000001FF,
  0x00000000000003FF,
  0x00000000000007FF,
  0x0000000000000FFF,
  0x0000000000001FFF,
  0x0000000000003FFF,
  0x0000000000007FFF,
  0x000000000000FFFF,
  0x000000000001FFFF,
  0x000000000003FFFF,
  0x000000000007FFFF,
  0x00000000000FFFFF,
  0x00000000001FFFFF,
  0x00000000003FFFFF,
  0x00000000007FFFFF,
  0x0000000000FFFFFF,
  0x0000000001FFFFFF,
  0x0000000003FFFFFF,
  0x0000000007FFFFFF,
  0x000000000FFFFFFF,
  0x000000001FFFFFFF,
  0x000000003FFFFFFF,
  0x000000007FFFFFFF,
  0x00000000FFFFFFFF,
  0x00000001FFFFFFFF,
  0x00000003FFFFFFFF,
  0x00000007FFFFFFFF,
  0x0000000FFFFFFFFF,
  0x0000001FFFFFFFFF,
  0x0000003FFFFFFFFF,
  0x0000007FFFFFFFFF,
  0x000000FFFFFFFFFF,
  0x000001FFFFFFFFFF,
  0x000003FFFFFFFFFF,
  0x000007FFFFFFFFFF,
  0x00000FFFFFFFFFFF,
  0x00001FFFFFFFFFFF,
  0x00003FFFFFFFFFFF,
  0x00007FFFFFFFFFFF,
  0x0000FFFFFFFFFFFF,
  0x0001FFFFFFFFFFFF,
  0x0003FFFFFFFFFFFF,
  0x0007FFFFFFFFFFFF,
  0x000FFFFFFFFFFFFF,
  0x001FFFFFFFFFFFFF,
  0x003FFFFFFFFFFFFF,
  0x007FFFFFFFFFFFFF,
  0x00FFFFFFFFFFFFFF,
  0x01FFFFFFFFFFFFFF,
  0x03FFFFFFFFFFFFFF,
  0x07FFFFFFFFFFFFFF,
  0x0FFFFFFFFFFFFFFF,
  0x1FFFFFFFFFFFFFFF,
  0x3FFFFFFFFFFFFFFF,
  0x7FFFFFFFFFFFFFFF,
  0xFFFFFFFFFFFFFFFF
};


#ifdef __CUDACC__ // if compiled with nvcc

//64-bits mask for the GPU device (65 elements from 0 to 64 included)
__device__ unsigned long long int ulln_dmasks64[65] = {
  0x0000000000000000,
  0x0000000000000001,
  0x0000000000000003,
  0x0000000000000007,
  0x000000000000000F,
  0x000000000000001F,
  0x000000000000003F,
  0x000000000000007F,
  0x00000000000000FF,
  0x00000000000001FF,
  0x00000000000003FF,
  0x00000000000007FF,
  0x0000000000000FFF,
  0x0000000000001FFF,
  0x0000000000003FFF,
  0x0000000000007FFF,
  0x000000000000FFFF,
  0x000000000001FFFF,
  0x000000000003FFFF,
  0x000000000007FFFF,
  0x00000000000FFFFF,
  0x00000000001FFFFF,
  0x00000000003FFFFF,
  0x00000000007FFFFF,
  0x0000000000FFFFFF,
  0x0000000001FFFFFF,
  0x0000000003FFFFFF,
  0x0000000007FFFFFF,
  0x000000000FFFFFFF,
  0x000000001FFFFFFF,
  0x000000003FFFFFFF,
  0x000000007FFFFFFF,
  0x00000000FFFFFFFF,
  0x00000001FFFFFFFF,
  0x00000003FFFFFFFF,
  0x00000007FFFFFFFF,
  0x0000000FFFFFFFFF,
  0x0000001FFFFFFFFF,
  0x0000003FFFFFFFFF,
  0x0000007FFFFFFFFF,
  0x000000FFFFFFFFFF,
  0x000001FFFFFFFFFF,
  0x000003FFFFFFFFFF,
  0x000007FFFFFFFFFF,
  0x00000FFFFFFFFFFF,
  0x00001FFFFFFFFFFF,
  0x00003FFFFFFFFFFF,
  0x00007FFFFFFFFFFF,
  0x0000FFFFFFFFFFFF,
  0x0001FFFFFFFFFFFF,
  0x0003FFFFFFFFFFFF,
  0x0007FFFFFFFFFFFF,
  0x000FFFFFFFFFFFFF,
  0x001FFFFFFFFFFFFF,
  0x003FFFFFFFFFFFFF,
  0x007FFFFFFFFFFFFF,
  0x00FFFFFFFFFFFFFF,
  0x01FFFFFFFFFFFFFF,
  0x03FFFFFFFFFFFFFF,
  0x07FFFFFFFFFFFFFF,
  0x0FFFFFFFFFFFFFFF,
  0x1FFFFFFFFFFFFFFF,
  0x3FFFFFFFFFFFFFFF,
  0x7FFFFFFFFFFFFFFF,
  0xFFFFFFFFFFFFFFFF
};


//Computes (a*x+b) mod 256 on the GPU with long long integers and better accuracy
//input : d_a is the a in the expression (a*x+b)%256
//input : ulln_x is the x in t he expression (a*x+b)%256
//input : d_b is the b in the expression (a*x+b)%256
//output : returns (a*x+b)%256
__device__ double dbrad_modlli64d(double d_a, unsigned long long ulln_x, double d_b) {

    UDoubleBit udb_a, udb_b;
    udb_a.d = d_a;
    udb_b.d = d_b;

    //Extract exponents
    int n_expa = ((udb_a.uln_d & 0x7FF0000000000000)>>52) -1023;
    int n_expb = ((udb_b.uln_d & 0x7FF0000000000000)>>52) -1023;
    int n_exp = n_expa;

    //Extract mantissa
    unsigned long long int ulln_mantissa_a = udb_a.uln_d & 0x000FFFFFFFFFFFFF;
    unsigned long long int ulln_mantissa_b = udb_b.uln_d & 0x000FFFFFFFFFFFFF;

    int ndiff = 0;
    int ndiff_a = 0;
    int ndiff_b = 0;

    unsigned long long int ulln_a_with1, ulln_b_with1;

    //Shift bits to align the lowest value with the greatest

    if(n_expa!=-1023 && n_expb!=-1023){

      ndiff = n_expa - n_expb;

      if (ndiff > 0) {
        ulln_mantissa_a <<= ndiff;
        n_expb += ndiff;
        ndiff_a = ndiff;
        ndiff_b = 0;
      } else if (ndiff < 0) {
        ulln_mantissa_b <<= abs(ndiff);
        n_expa += abs(ndiff);
        ndiff_a = 0;
        ndiff_b = abs(ndiff);
      }
   }

  //Add the implicit most significant bit to the mantissa
  ulln_a_with1 = ulln_mantissa_a | (1ULL << (52+ndiff_a));
  ulln_b_with1 = ulln_mantissa_b | (1ULL << (52+ndiff_b));

  //If a==0 then exponent(a)==-1023
  if(n_expa==-1023){
    ulln_a_with1=0;
    n_exp = n_expb;
  }

  //If b==0 then exponent(b)==-1023
  if(n_expb==-1023){
    ulln_b_with1=0;
    n_exp = n_expa;
  }

  //Do main calculus
  unsigned long long int ulln_a_times_x_plus_b = ulln_a_with1 * ulln_x + ulln_b_with1;

  //Compute position of the comma from the right
  int ncomma_pos_from_right = 52 - n_exp + (ndiff>0 ? ndiff : 0);
  //Apply mask for 256 modulus
  unsigned un_digits_mask = ncomma_pos_from_right + 8;
  unsigned long long int ulln_mantissa = ulln_a_times_x_plus_b & ulln_dmasks64[64<un_digits_mask ? 64 : un_digits_mask];

  //return final result
  return (ulln_mantissa / (double)(1ULL << ncomma_pos_from_right));

}


#endif




//Computes (a*x+b) mod 256 on the CPU with long long integers and better accuracy
//input : d_a is the a in the expression (a*x+b)%256
//input : ulln_x is the x in t he expression (a*x+b)%256
//input : d_b is the b in the expression (a*x+b)%256
//output : returns (a*x+b)%256
double hbrad_modlli64d(double d_a, unsigned long long ulln_x, double d_b) {

    UDoubleBit udb_a, udb_b;
    udb_a.d = d_a;
    udb_b.d = d_b;

    //Extract exponents
    int n_expa = ((udb_a.uln_d & 0x7FF0000000000000)>>52) -1023;
    int n_expb = ((udb_b.uln_d & 0x7FF0000000000000)>>52) -1023;
    int n_exp = n_expa;

    //Extract mantissa
    unsigned long long int ulln_mantissa_a = udb_a.uln_d & 0x000FFFFFFFFFFFFF;
    unsigned long long int ulln_mantissa_b = udb_b.uln_d & 0x000FFFFFFFFFFFFF;

    int ndiff = 0;
    int ndiff_a = 0;
    int ndiff_b = 0;

    //Shift bits to align the lowest value with the greatest

    unsigned long long int ulln_a_with1, ulln_b_with1;

    if(n_expa!=-1023 && n_expb!=-1023){

      ndiff = n_expa - n_expb;

      if (ndiff > 0) {
        ulln_mantissa_a <<= ndiff;
        n_expb += ndiff;
        ndiff_a = ndiff;
        ndiff_b = 0;
      } else if (ndiff < 0) {
        ulln_mantissa_b <<= abs(ndiff);
        n_expa += abs(ndiff);
        ndiff_a = 0;
        ndiff_b = abs(ndiff);
      }
   }

  //Add the implicit most significant bit to the mantissa
  ulln_a_with1 = ulln_mantissa_a | (1ULL << (52+ndiff_a));
  ulln_b_with1 = ulln_mantissa_b | (1ULL << (52+ndiff_b));

  //If a==0 then exponent(a)==-1023
  if(n_expa==-1023){
    ulln_a_with1=0;
    n_exp = n_expb;
  }

  //If b==0 then exponent(b)==-1023
  if(n_expb==-1023){
    ulln_b_with1=0;
    n_exp = n_expa;
  }

  //Do main calculus
  unsigned long long int ulln_a_times_x_plus_b = ulln_a_with1 * ulln_x + ulln_b_with1;

  //Compute position of the comma from the right
  int ncomma_pos_from_right = 52 - n_exp + (ndiff>0 ? ndiff : 0);

  //Apply mask for modulo 256
  unsigned un_digits_mask = ncomma_pos_from_right + 8;
  unsigned long long int ulln_mantissa = ulln_a_times_x_plus_b & ulln_hmasks64[64<un_digits_mask ? 64 : un_digits_mask];

  //Return final value
  return (ulln_mantissa / (double)(1ULL << ncomma_pos_from_right));

}
