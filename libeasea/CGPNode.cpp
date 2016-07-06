#include "include/CGPNode.h"
#include "include/CRandomGenerator.h"
#include <stdlib.h>
#include <string.h>

#include<iostream>
#include<sstream>


extern CRandomGenerator* globalRandomGenerator;
extern unsigned opArity[];
//extern const unsigned* opArity;


/**
   Compute the maximum depth of a tree, rooted on root.

   @arg root : root of the tree
   @return : depth of current tree rooted on root
*/
int depthOfTree(GPNode* root) {
    int depth = 0;
    for( unsigned i=0 ; i<opArity[(int)root->opCode] ; i++ ) {
        int d = depthOfTree(root->children[i]);
        if( d>=depth ) depth = d;
    }
    return depth+1;
}


int depthOfNode(GPNode* root, GPNode* node) {

    if( root==node ) {
        return 1;
    } else {
        for( unsigned i=0 ; i<opArity[(int)root->opCode] ; i++ ) {
            int depth = depthOfNode(root->children[i],node);
            if( depth )
                return depth+1;
        }
        return 0;
    }
}


int enumTreeNodes(GPNode* root) {
    int nbNode = 0;
    for( unsigned i=0 ; i<opArity[(int)root->opCode] ; i++ ) {
        nbNode+=enumTreeNodes(root->children[i]);
    }
    return nbNode+1;
}




void flattenDatas2D( float** inputs, int length, int width,
                     float** flat_inputs) {
    (*flat_inputs)=(float*)malloc(sizeof(float)*length*width);
    for( int i=0 ; i<length ; i++ ) {
        memcpy( (*flat_inputs)+(i*width),inputs[i],width*sizeof(float));
    }
}


/**
   Fill the collection array with GPNode located at goalDepth

   @arg goalDepth: level from which GPNode are collected
   @arg collection: an empty, allocated array
*/
int collectNodesDepth(const int goalDepth, GPNode** collection, int collected,
                      int currentDepth, GPNode* root)\
{

    if( currentDepth>=goalDepth ) {
        collection[collected] = root;
        return collected+1;
    } else{
        for( unsigned i=0 ; i<opArity[(int)root->opCode] ; i++ ) {
            collected=collectNodesDepth(goalDepth, collection, collected, currentDepth+1,
            root->children[i]);
        }
        return collected;
    }
}

/**
   Pick a node in a tree. It first pick a depth and then, it pick a
   node amongst nodes at this depth. It returns the parent node,
   and by pointer, the childId of the choosen child.

   @arg root : the root node of the tree, amongt which we have to choose the node
   @arg chilId : pointer to an allocated int, children position of the choosen node will be stored here
   @arg depth : pointer to an allocated int, will contain the choosen depth.

   @return : return the address of the parent of the choosen node. Return null if the root node has been choos\
en
*/
GPNode* selectNode( GPNode* root, int* childId, int* depth) {

    int xoverDepth = globalRandomGenerator->random(0,depthOfTree(root));
    (*depth) = xoverDepth;

    GPNode** dNodes;
    int collected;

    if(xoverDepth!=0) {
        dNodes = new GPNode*[1<<(xoverDepth-1)];
        collected = collectNodesDepth(xoverDepth-1,dNodes,0,0,root);
    } else {
        return NULL;
    }
    int stockPointCount=0;
    for( int i=0 ; i<collected; i++ ) {
        stockPointCount+=opArity[(int)dNodes[i]->opCode];
    }

    int reminderP = 0, parentIndexP = 0;

    unsigned xoverP = globalRandomGenerator->random(0,stockPointCount);
    for( unsigned i=0 ; ; )
        if( (i+opArity[(int)dNodes[parentIndexP]->opCode])>xoverP ) {
            reminderP = xoverP-i;
            break;
        } else i+=opArity[(int)dNodes[parentIndexP++]->opCode];

    *childId = reminderP;
    //cout << "d of x : " << xoverDepth << "/" << depthOfTree(root)<< " n : "<< xoverP << endl;
    GPNode* ret = dNodes[parentIndexP];
    delete[] dNodes;
    return ret;
}



/**
   Recursive construction method for trees.
   Koza construction methods. Function set has to be ordered,
   with first every terminal nodes and then non-terminal.

   @arg constLen : length of terminal function set.
   @arg totalLen : length of the function set (non-terminal+terminal)
   @arg currentDepth : depth of the origin (sould always be 0, when the function
   is directly call)
   @arg maxDepth : The maximum depth of the resulting tree.
   @arg full : whether the construction method used has to be full (from koza's book)
   Otherwise, it will use grow method (defined in the same book).

   @return : pointer to the root node of the resulting sub tree
*/
GPNode* construction_method( const int constLen, const int totalLen ,
                             const int currentDepth,
                             const int maxDepth, const bool full,
                             const unsigned* opArity, const int OP_ERC) {
    GPNode* node = new GPNode();
    // first select the opCode for the current Node.
    if( full ) {
        if( currentDepth<maxDepth ) node->opCode = globalRandomGenerator->random(
                        constLen,totalLen);
        else node->opCode = globalRandomGenerator->random(0, constLen);
    } else {
        if( currentDepth<maxDepth ) node->opCode = globalRandomGenerator->random(0,
                    totalLen);
        else node->opCode = globalRandomGenerator->random(0, constLen);
    }

    int arity = opArity[(int)node->opCode];
    //node->arity = arity;

    // construct children (if any)
    for( int i=0 ; i<arity ; i++ )
        node->children[i] = construction_method(constLen, totalLen, currentDepth+1,
                                                maxDepth, full, opArity, OP_ERC);

    // affect null to other array cells (if any)
    for( int i=arity ; i<MAX_ARITY ; i++ )
        node->children[i] = NULL;

    if( node->opCode==OP_ERC ) {
        node->erc_value = globalRandomGenerator->random(0.,1.);
    }
    //else if( node->opCode==OP_VAR )
    //node->var_id = globalRandomGenerator->random(1,VAR_LEN);

    return node;
}

GPNode* RAMPED_H_H(unsigned INIT_TREE_DEPTH_MIN, unsigned INIT_TREE_DEPTH_MAX,
                   unsigned actualParentPopulationSize, unsigned parentPopulationSize,
                   float GROW_FULL_RATIO, unsigned VAR_LEN, unsigned OPCODE_SIZE,
                   const unsigned* opArity, const int OP_ERC) {
    /**
       This is the standard ramped half-and-half method
       for creation of trees.
     */
    int id = actualParentPopulationSize;
    int seg = parentPopulationSize/(INIT_TREE_DEPTH_MAX-INIT_TREE_DEPTH_MIN);
    int currentDepth = INIT_TREE_DEPTH_MIN+id/seg;

    bool full;
    if( GROW_FULL_RATIO==0 ) full=true;
    else full = (id%seg)/(int)(seg*GROW_FULL_RATIO);

    //cout << seg << " " <<  currentDepth << " " << full ;
    return construction_method( VAR_LEN+1, OPCODE_SIZE , 1, currentDepth ,full,
                                opArity, OP_ERC);
}

void toString_r(std::ostringstream* oss, GPNode* root, const unsigned* opArity ,
                const char** opCodeName, int OP_ERC) {

    (*oss) << '(';
    if (opArity[(int)root->opCode] == 2) {
        toString_r(oss,root->children[0],opArity,opCodeName,OP_ERC);
        (*oss) << ' ';
        (*oss) << opCodeName[(int)root->opCode];
        (*oss) << ' ';
        toString_r(oss,root->children[1],opArity,opCodeName,OP_ERC);
    } else {
        if (root->opCode == OP_ERC) {
            (*oss) << root->erc_value;
        } else {
            (*oss) << opCodeName[(int)root->opCode];
        }
        for (unsigned i = 0; i < opArity[(int)root->opCode]; ++i) {
            if (root->children[i]) {
                toString_r(oss,root->children[i],opArity,opCodeName,OP_ERC);
                if (i < opArity[(int)root->opCode] - 1) {
                    (*oss) << ' ';
                }
            }
        }
    }
    (*oss) <<  ')';

    return;
}

std::string toString(GPNode* root, const unsigned* opArity ,
                     const char** opCodeName, int OP_ERC) {
    std::ostringstream oss;

    toString_r(&oss,root,opArity,opCodeName,OP_ERC);

    return oss.str();
}


/**
   This function handles printing of tree.
   Every node is identify by its address, in memory,
   and labeled by the actual opCode.

   On our architecture (64bits, ubuntu 8.04 and gcc-4.3.2)
   the long int variable is sufficient to store the address
   without any warning.
 */
void toDotFile_r(GPNode* root, FILE* outputFile, const unsigned* opArity ,
                 const char** opCodeName, int OP_ERC) {
    if( root->opCode==OP_ERC )
        fprintf(outputFile," %ld [label=\"%s : %f\"];\n", (long int)root,
                opCodeName[(int)root->opCode],
                root->erc_value);
    else
        fprintf(outputFile," %ld [label=\"%s\"];\n", (long int)root,
                opCodeName[(int)root->opCode]);

    for( unsigned i=0 ; i<opArity[(int)root->opCode] ; i++ ) {
        if( root->children[i] ) {
            fprintf(outputFile,"%ld -> %ld;\n", (long int)root,
                    (long int)root->children[i]);
            toDotFile_r( root->children[i] , outputFile,opArity,opCodeName,OP_ERC);
        }
    }
}




/**
   This function prints a tree in dot (graphviz format).
   This is the entry point for the print operation. (see toDotFile_r,
   for the actual function)

   @arg root : set of trees, same type than in a individual.
   @arg baseFileName : base of filename for the output file.
   @arg treeId : the id of the tree to print, in the given set.
 */
void toDotFile(GPNode* root, const char* baseFileName, int treeId,
               const unsigned* opArity , const char** opCodeName, int OP_ERC) {
    std::ostringstream oss;
    oss << baseFileName << "-" << treeId << ".gv";

    FILE* outputFile = fopen(oss.str().c_str(),"w");
    if( !outputFile ) {
        perror("Opening file for outputing dot representation ");
        printf("%s\n",oss.str().c_str());
        exit(-1);
    }

    fprintf(outputFile,"digraph trees {\n");
    if(root)
        toDotFile_r( root, outputFile,opArity,opCodeName,OP_ERC);
    fprintf(outputFile,"}\n");
    fclose(outputFile);
}
