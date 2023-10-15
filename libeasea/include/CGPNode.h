/**
 * @file CGPNode.h
 * @version 1.0
 *
 **/  

#ifndef EZ_C_GPNODE
#define EZ_C_GPNODE

#include <iostream>

#define MAX_ARITY 2          // maximum arrity for GP node

/**
 *  \class   GPNode 
 *  \brief   Genetic Programming 
 *  \details Used to modelised nodes of abstract syntax tree
 *  
 **/ 

class GPNode {
  public:


    GPNode(){  // Constructor
      for(int EASEA_Ndx=0; EASEA_Ndx<2; EASEA_Ndx++)
        children[EASEA_Ndx]=NULL;
    }


    GPNode(int var_id, double erc_value, char opCode, GPNode** childrenToAdd) : var_id(var_id), erc_value(erc_value), opCode(opCode)// other constructor
  {
    for(int EASEA_Ndx=0; EASEA_Ndx<2; EASEA_Ndx++)
      this->children[EASEA_Ndx]=childrenToAdd[EASEA_Ndx];
  }  


    GPNode(const GPNode &EASEA_Var) : GPNode() {  // Copy constructor
      var_id=EASEA_Var.var_id;
      erc_value=EASEA_Var.erc_value;
      //arity=EASEA_Var.arity;
      opCode=EASEA_Var.opCode;
      
      for(int EASEA_Ndx=0; EASEA_Ndx<2; EASEA_Ndx++)
	if( EASEA_Var.children[EASEA_Ndx] ) children[EASEA_Ndx] = new GPNode(*(EASEA_Var.children[EASEA_Ndx]));
    }


    virtual ~GPNode() {  // Destructor
      for(int EASEA_Ndx=0; EASEA_Ndx<2; EASEA_Ndx++)
        if( children[EASEA_Ndx] ) delete children[EASEA_Ndx];
    }


    GPNode& operator=(const GPNode &EASEA_Var) {  // Operator=
      if (&EASEA_Var == this) return *this;
      var_id = EASEA_Var.var_id;
      erc_value = EASEA_Var.erc_value;
      //arity = EASEA_Var.arity;
      opCode = EASEA_Var.opCode;
      
      for(int EASEA_Ndx=0; EASEA_Ndx<2; EASEA_Ndx++)
        if(EASEA_Var.children[EASEA_Ndx]) {
		delete children[EASEA_Ndx];
		children[EASEA_Ndx] = new GPNode(*(EASEA_Var.children[EASEA_Ndx]));
	}
      
      return *this;
    }


    bool operator==(GPNode &EASEA_Var) const {  // Operator==
      if (var_id!=EASEA_Var.var_id) return false;
      if (erc_value!=EASEA_Var.erc_value) return false;
      //if (arity!=EASEA_Var.arity) return false;
      if (opCode!=EASEA_Var.opCode) return false;
      
      {for(int EASEA_Ndx=0; EASEA_Ndx<2; EASEA_Ndx++)
        if (children[EASEA_Ndx]!=EASEA_Var.children[EASEA_Ndx]) return false;}

      return true;
    }


    bool operator!=(GPNode &EASEA_Var) const {return !(*this==EASEA_Var);} // operator!=


    friend std::ostream& operator<< (std::ostream& os, const GPNode& EASEA_Var) { // Output stream insertion operator
      os <<  "var_id:" << EASEA_Var.var_id << "\n";
      os <<  "erc_value:" << EASEA_Var.erc_value << "\n";
      //os <<  "arity:" << EASEA_Var.arity << "\n";
      os <<  "opCode:" << EASEA_Var.opCode << "\n";
      
      {os << "Array children : ";
        for(int EASEA_Ndx=0; EASEA_Ndx<2; EASEA_Ndx++)
          if( EASEA_Var.children[EASEA_Ndx] ) os << "[" << EASEA_Ndx << "]:" << *(EASEA_Var.children[EASEA_Ndx]) << "\t";}
      
      os << "\n";

      return os;
    }

    template <class Archive>
    void serialize(Archive & ar, [[maybe_unused]] const unsigned int version) {
	    ar & var_id;
	    ar & erc_value;
	    ar & opCode;
	    ar & children;
    }


    // Class members 
    int var_id;
    double erc_value;
    // char opCode;
    int opCode;
    GPNode* children[2];
};

/* Here are some utility functions for the template GP */
int depthOfTree(GPNode* root, const unsigned opArity[]);
int enumTreeNodes(GPNode* root, const unsigned opArity[]);
int depthOfNode(GPNode* root, GPNode* node, const unsigned opArity[]);

//void flattenDatas( float** inputs, int length, int width, float** flat_inputs);
GPNode* selectNode( GPNode* root, int* childId, int* depth, const unsigned opArity[]);
GPNode* RAMPED_H_H(unsigned iINIT_TREE_DEPTH_MIN, unsigned iINIT_TREE_DEPTH_MAX, unsigned actualParentPopulationSize, unsigned parentPopulationSize, float iGROW_FULL_RATIO, unsigned iVAR_LEN, unsigned iOPCODE_SIZE, const unsigned opArity[], const int iOP_ERC);
void flattenDatas2D( float** inputs, int length, int width, float** flat_inputs);

GPNode* construction_method( const int constLen, const int totalLen , const int currentDepth, const int maxDepth, const bool full, const unsigned opArity[], const int OP_ERC);

// display methods
void toDotFile(GPNode* root, const char* baseFileName, int treeId, const unsigned opArity[] , const char** opCodeName, int OP_ERC);
std::string toString(GPNode* root, const unsigned opArity[] , const char* const* opCodeName, int OP_ERC);

#endif // __C_GPNODE__
