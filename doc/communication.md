# Piggybacking specification

## IP type
 IPv4 or IPv6

## Empty
  
  Specify if the body is empty.
  
## pType

### As an enum with their corresponding value:

* XOVER              [float32]
* MUT                [float32]
* PRESSURE_PARENT    [float32]
* PRESSURE_OFFSPRING [float32]
* PRESSURE_FINAL     [float32]
* MIGRATION          [float32]
* IPV4_ADD           [128 bits ]
* IPV4_REMOVE        [128 bits ]
* IPV6_ADD           [512 bits ]
* IPV6_REMOVE        [512 bits ]
* IP_REMOVE_ALL      [No data]
* STOP               [No data]

* IPv4 are sent as 4 aligned unsigned char, and IPv6 16.
* All other parameters are probabilities between 0 and 1.
* IP operations are done in the order of the header. If removeAll is set, it
 take precedence of every operations on IP list.

## Methods

The main methods will be:
* packetHeader* createHeader( float xover=-1,float mut=-1,float
                              pressureParent=-1, float pressureOffspring=-1, 
                              float pressureFinal=-1, float migration=-1, 
                              bool removeAll=false)
* addIPHeader(packetHeader* header, bool ipv6,char* ip,bool remove)

* create_packet(packetHeader* ,char*)
* process_packet(void* rawData)
* process_header(void* rawData)

