
#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <istream>
#include <ostream>
#include <string>
#include <initializer_list>
#ifndef Abs
#define Abs(x) ((x)>=0?(x):-(x))
#endif
template <class T>
class CMatrix
{
private:
    template <typename TE> friend CMatrix<TE> operator*(const CMatrix<TE>& a, const CMatrix<TE>& b);
    template <typename TE> friend CMatrix<TE> operator*(const TE& a, const CMatrix<TE>& b);
    template <typename TE> friend CMatrix<TE> operator*(const CMatrix<TE>& a, const TE& b);
    template <typename TE> friend CMatrix<TE> operator+(const CMatrix<TE>& a, const CMatrix<TE>& b);
    template <typename TE> friend CMatrix<TE> operator-(const CMatrix<TE>& a, const CMatrix<TE>& b);
    template <typename TE> friend bool operator==(const CMatrix<TE>& a, const CMatrix<TE>& b);
    template <typename TE> friend std::ostream& operator<<(std::ostream &os, const CMatrix<TE>& x);
    template <typename TE> friend std::istream& operator>>(std::istream &is, CMatrix<TE>& x);

public :
    std::vector<std::vector<T>> mat;
private :
    unsigned N;
    unsigned M;
    int error;


public:
    CMatrix<T>(): N(0), M(0) {}

    CMatrix<T>(unsigned N, unsigned M, T init=0)
    {
        this->N=N;
        this->M=M;

        mat.resize(N);
        for (unsigned i=0; i<N; i++)
        {
            mat[i].resize(M);
            fill(mat[i].begin(),mat[i].end(),init);
        }

    }

    CMatrix<T>(const CMatrix &m)
    {
        mat=m.mat;
        N=m.N;
        M=m.M;
    }
    CMatrix<T>(const std::vector<std::vector<T>> &m)
    {
        unsigned c=0;
        if(m.size()>0)
        {
            c=m[0].size();
            for (unsigned i=1; i<m.size();i++)
                if(m[i].size()!=c)
                    throw std::logic_error("Not a CMatrix");
        }
        mat=m;
        N=m.size();
        M=c;
    }
    CMatrix<T>(std::initializer_list<std::initializer_list<T>> list)
    {
         N=list.size();
         mat.resize(N);

         M=0;
         if (N>0)
             M=list.begin()->size();

         unsigned i=0;
         for (const auto& r : list)
         {
             mat[i].resize(M);
             unsigned j=0;
             for (const auto& val : r)
             {
                 mat[i][j]=val;;
                 j++;
             }
             if (M!=j)
                 throw std::logic_error("Not a CMatrix");

             i++;

         }
    }
        CMatrix<T> CholeskyInverse();
    CMatrix<T> Cholesky(T ztol);
    CMatrix<T> inverse1();

    CMatrix<T>& operator=(const std::vector<std::vector<T>> &m);
    std::vector<T>& operator[](unsigned i) { return mat[i]; }
    std::vector<T> operator[](unsigned i) const { return mat[i]; }
    CMatrix<T>& operator+=(const CMatrix<T> &m);
    CMatrix<T>& operator-=(const CMatrix<T> &m);
    CMatrix<T>& operator*=(const CMatrix<T> &m);

    CMatrix<T>& Fill(const T &val)
    {
        for (unsigned i=0; i<N; i++)
            fill(mat[i].begin(),mat[i].end(),val);

        return *this;
    }

    CMatrix<T>& AppendRow(T init=0)
    {
        std::vector<T> r(M,init);
        mat.push_back(r);
        N++;
        return *this;
    }
    CMatrix<T>& AppendCol(T init=0)
    {
        for (unsigned i=0;i<N;i++)
            mat[i].push_back(init);
        M++;
        return *this;
    }

    CMatrix<T>& Clear()
    {
        mat.clear();
        N=M=0;
        return *this;
    }


    unsigned Rows() const { return N; }
    unsigned Cols() const { return M; }

    CMatrix<T> Coff(unsigned i, unsigned j) const;
    T Det() const;
    CMatrix<T> Inverse() const;
    static T Det(const CMatrix<T>& x);
    static CMatrix<T> LinSolve(const CMatrix &A, const CMatrix &b) { return A.Inv()*b; }


};
template <typename TE>
CMatrix<TE> operator*(const CMatrix<TE>& a, const CMatrix<TE>& b)
{
    if(a.M != b.N)
        throw std::logic_error("columns(A)!=rows(B)");

    if(a.N<1)
        throw std::logic_error("Empty matrices");

    CMatrix<TE> y(a.N,b.M);
    for(unsigned i=0;i<y.N; i++)
    {
        for(unsigned j=0; j<y.M; j++)
        {
            TE sum=0;
            for(unsigned k=0; k<b.N; k++)
                sum+=a.mat[i][k]*b.mat[k][j];

            y.mat[i][j]=sum;

        }
    }

    return y;
}       

template <typename TE>
CMatrix<TE> operator*(const TE& a, const CMatrix<TE>& b)
{
    CMatrix<TE> y(b.N,b.M);
    for (unsigned i=0; i< b.N; i++)
        for (unsigned j=0; j<b.M; j++)
            y[i][j]=a*b[i][j];

    return y;
}
template <typename TE>
CMatrix<TE> operator+(const CMatrix<TE>& a, const CMatrix<TE>& b)
{
    if (a.N != b.N || a.M!= b.M)
        throw std::logic_error("Operator+ requires two matrices of same size");
    
    CMatrix<TE> y(a.N,a.M);
    for (unsigned i=0; i<a.N; i++)
        for (unsigned j=0; j<a.M; j++)
            y.mat[i][j]=a.mat[i][j]+b.mat[i][j];

    return y;
}


template <typename TE>
CMatrix<TE> operator-(const CMatrix<TE>& a, const CMatrix<TE>& b)
{
    if (a.N != b.N || a.M!= b.M)
        throw std::logic_error("Operator- requires two matrices of same size");

    CMatrix<TE> y(a.N,a.M);
    for (unsigned i=0; i<a.N; i++)
        for (unsigned j=0; j<a.M; j++)
            y.mat[i][j]=a.mat[i][j]-b.mat[i][j];

    return y;
}
template <typename TE>
CMatrix<TE> operator*(const CMatrix<TE>& a, const TE& b)
{
    return b*a;
}

template <typename TE>
bool operator==(const CMatrix<TE>& a, const CMatrix<TE>& b)
{
    if (a.N != b.N || a.M |= b.M)
        return false;

    for (unsigned i=0; i<a.N; i++)
        if(a[i]!=b[i])
            return false;

    return true;
}

    
template <typename TE>
std::ostream& operator<<(std::ostream &os, const CMatrix<TE>& x)
{
    os << '[' << std::endl;
    for (unsigned i=0; i<x.N-1; i++)
    {
        for (unsigned j=0; j<x.M-1; j++)
            os << x.mat[i][j] << ",\t";

        os << x.mat[i][x.M-1] << " ;" << std::endl;
    }

    if(x.N>0)
    {
        for (unsigned j=0; j<x.M-1; j++)
                os << x.mat[x.N-1][j] << ",\t";

        os << x.mat[x.N-1][x.M-1] << std::endl;
    }

    os << "]" <<std::endl;

    return os;
}
 
template <typename TE>
std::istream& operator>>(std::istream &is, CMatrix<TE>& x)
{
    unsigned m;
    x.Clear();
    char c;
    TE v;

    is>>c;
    int t = is.peek();
    if (t == std::istream::traits_type::eof())
	    throw std::logic_error("Bad input stream");
    c = static_cast<char>(t);

    if (c!=']')
    {
        do
        {
            x.AppendRow();
            m=0;
            c=0;
            while (c != ';' && c != ']')
            {
                is>>v>>c;
                m++;
                if(m>x.M && x.N>1)
                    throw std::logic_error("Input is not a mtrix");

                if(m>x.M)
                    x.AppendCol();
		x.mat[x.N-1][m-1]=v;

            }
        } while(c != ']');

        if(m != x.M)
            throw std::logic_error("Input is not a CMatrix");
    }
            
    return is;

}

// CLASS METHODS
template <class T>
CMatrix<T>& CMatrix<T>::operator=(const std::vector<std::vector<T>> &m)
{
    unsigned c=0;
    if(m.size()>0)
    {
        c=m[0].size();
        for (unsigned i=1; i<m.size();i++)
            if(m[i].size()!=c)
                throw std::logic_error("Not a CMatrix");
    }
    mat=m.mat;
    N=m.size();
    M=c;

    return *this;
}
template <class T>
CMatrix<T>& CMatrix<T>::operator+=(const CMatrix<T> &m)
{
    if (N != m.N || M!= m.M)
        throw std::logic_error("Operator+= requires two matrices of same size");

    for (unsigned i=0; i<N; i++)
        for (unsigned j=0; j<M; j++)
            mat[i][j]+=m.mat[i][j];

    return *this;
}

template <class T>
CMatrix<T>& CMatrix<T>::operator-=(const CMatrix<T> &m)
{
    if (N != m.N || M != m.M)
        throw std::logic_error("Operator= requires two matrices of same size");

    for (unsigned i=0; i<N; i++)
        for (unsigned j=0; j<M; j++)
            mat[i][j]-=m.mat[i][j];
    
    return *this;
}
template <class T>
CMatrix<T>& CMatrix<T>::operator*=(const CMatrix<T> &m)
{
    *this=*this*m;
    return *this;
}
template <class T>
CMatrix<T> CMatrix<T>::Coff(unsigned i, unsigned j) const
{
    if (N==0)
        throw std::logic_error("Coff: the CMatrix is empty");
    
    CMatrix<T> y(N-1,M-1);

    unsigned k_c=0;
    for (unsigned k_x=0; k_x<N;k_x++)
    {
        if (k_x==i)
            continue;

        unsigned j_c=0;
        for (unsigned j_x=0 ; j_x<N; j_x++)
        {
            if (j_x==j)
                continue;

            y.mat[k_c][j_c]=mat[k_x][j_x];
            j_c++;
        }

        k_c++;
    }

    return y;

}
template <class T>
T CMatrix<T>::Det(const CMatrix<T>& x)
{
    if(x.N != x.M)
        throw std::logic_error("Can't compute the determinant of a non square CMatrix");

    if (x.N == 0)
        throw std::logic_error("Empty CMatrix");

    if (x.N==1)
        return x.mat[0][0];

    T y=0;
    int d=1;

    for (unsigned i=0; i<x.N; i++)
    {
        y+=d*x.mat[i][0]*Det(x.Coff(i,0));
        d=-d;
    }

    return y;
}

template <class T>
T CMatrix<T>::Det() const
{
    return Det(*this);
}
template <class T>
CMatrix<T> CMatrix<T>::Inverse() const
{
    double det_x=Det();
    if(abs(det_x)<std::numeric_limits<double>::epsilon())
        throw std::logic_error("Can't invert CMatrix  (determinant=0)");

    CMatrix<T> y(N,M);

    signed int d=1;

    for (unsigned i=0; i<N; i++)
        for (unsigned j=0; j<N; j++)
        {
            y.mat[j][i]=d*Det(Coff(i,j))/det_x;
            d=-d;
        }

    return y;

}





