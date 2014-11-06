/**
 * @file CMonitorUtils.hpp
 * @author Pallamidessi Joseph
 * @version 1.0
 *
 * @section LICENSE
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details at
 * http://www.gnu.org/copyleft/gpl.html
**/

#ifndef AUDIOMONITORUTILS_H__
#define AUDIOMONITORUTILS_H__

#include <iostream>
#include <cstring>
#include <cstdio>
#include <cstdlib>

/*Use by the server to cast to the right type*/
enum{NOTUSE,SIMPLEDATA};

/**
*\struct    MonitorParameter 
*\brief     Abstract class that define the parameter structureand calls used by 
*           the audio monitoring system.
*\details   This class provided the call to serialization/deserialization, 
*           data processing and simulation specific method (getTime).
**/
class MonitorParameter {
  public:
    /*Constructors/Destructor-----------------------------------------------------*/
    /**
    * \brief    Constructor of MonitorParameter.
    * \details  Initialize all data fields.
    *
    **/
    MonitorParameter ();
    
    
    /**
    * \brief    Destructor of MonitorParameter.
    * \details 
    *
    **/
    virtual ~MonitorParameter();


    /*Methods---------------------------------------------------------------------*/
    /**
    * \brief    Fill the data fields.
    * \details  To override.
    *
    **/
    virtual void fill();
    
    
    /**
    * \brief    Set to notify a sending.
    * \details  Format the policy boolean to notify a sending
    *
    **/
    virtual void sending();
    
    
    /**
    * \brief    Set to notify a reception.
    * \details  Format the policy boolean to notify a sending
    *
    **/
    virtual void reception();
    
    
    /**
    * \brief    The size in byte of this parameter.
    * \details  Unused, for debugging purpose. 
    *
    * @return   size_t The size in byte of this instance.
    **/
    virtual size_t size();
    

    /**
    * \brief    Serialize this instance.
    * \details  Return a pointer to the serialized data. The programmer must free
    *           this pointer after use.  
    *
    * @return   char* A pointer to serialized data in memory.
    **/
    virtual char* serialize()=0;
    

    /**
    * \brief    Deserialize data from given pointer, and filled this instance data
    *           fields.
    * \details  The pointer must point to a valid serialization of the same type,
    *           because otherwise the data fields will be filled with garbage.
    *
    *  @param   buf Pointer to serialized object in memory. 
    **/
    virtual void deserialize(char* buf)=0;


    /**
    * \brief    Size of the serialization of this instance
    * \details  Used to know how many bytes send to the server.
    *
    * @return   size Size of the resulting serialization in bytes.
    **/
    virtual int serialSize()=0;


    /**
    * \brief    Process data contained in the buffer (line)
    * \details  Used by the simulation software, kind of analogue to fill() in a
    *           easea program. Modify internal data with those from the buffer.
    *
    *  @param   line A string (.csv and .dat lines for example)
    **/
    virtual void processBuffer(std::string line)=0;


    /**
    * \brief    Return the difference between timeEnd and timeBegin in milliseconds.
    * \details  Result is expressed in milliseconds.Symbolize the time of a
    *           generation. Used by simulation software.
    *
    * @return   time Duration of a generation. 
    **/
    virtual int getTime();
    

    /*Getter----------------------------------------------------------------------*/
    /**
    * \brief    Return true if this instance represent a reception.
    * \details  Getter.
    *
    * @return   bool The reception flag.
    **/
    bool isReception();
    
    
    /**
    * \brief    Return true if this instance represent a sending.
    * \details  Getter.
    *
    * @return   bool The sending flag.
    **/
    bool isSending();
    
    
    /**
    * \brief    Return true if this instance represent datas.
    * \details  Getter.
    *
    * @return   bool the data flag.
    **/
    bool isData();
    

  public:
    /*Data------------------------------------------------------------------------*/
    unsigned char strType; // The server need to know how to cast this struct
    /*TODO: Better use a mask instead of four boolean ...*/
    bool dataFlag;    
    bool migration;
    bool recv;
    bool send;
    float timeBegin;
    float timeEnd;
    
};

#endif /* end of include guard: AUDIOMONITORUTILS_H__ */
