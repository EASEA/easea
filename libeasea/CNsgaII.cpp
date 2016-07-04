/********************************************************************/
/* NSGA-II routines 				     		    */
/* code readapated from Deb's original NSGA-II code version 1.1.6   */
/********************************************************************/


/* Memory allocation and deallocation routines */

# include "include/CIndividual.h"
# include <stdio.h>
# include <stdlib.h>
# include <vector>
# include <math.h>

namespace evo {

# define NSGA2_NOBJ 3
# define INF 1.0e14
# define EPS 1.0e-14
# define E  2.71828182845905
# define PI 3.14159265358979
# define GNUPLOT_COMMAND "gnuplot -persist"

typedef struct {
    int rank;
    double *obj;
    double crowd_dist;
    int id;
}
individual;

typedef struct {
    individual *ind;
}
NSGA_population;

typedef struct listNSGAs {
    int index;
    struct listNSGAs *parent;
    struct listNSGAs *child;
}
listNSGA;


void allocate_memory_pop (NSGA_population *pop, int size);
void allocate_memory_ind (individual *ind);
void deallocate_memory_pop (NSGA_population *pop, int size);
void deallocate_memory_ind (individual *ind);
void assign_crowding_distance_listNSGA (NSGA_population *pop, listNSGA *lst,
                                        int front_size);
void assign_crowding_distance_indices (NSGA_population *pop, int c1, int c2);
void assign_crowding_distance (NSGA_population *pop, int *dist, int **obj_array,
                               int front_size);
int check_dominance (individual *a, individual *b);
void insert (listNSGA *node, int x);
void quicksort_front_obj(NSGA_population *pop, int objcount, int obj_array[],
                         int obj_array_size);
void q_sort_front_obj(NSGA_population *pop, int objcount, int obj_array[],
                      int left, int right);
void quicksort_dist(NSGA_population *pop, int *dist, int front_size);
void q_sort_dist(NSGA_population *pop, int *dist, int left, int right);
void assign_rank_and_crowding_distance (NSGA_population *new_pop);
void advance_random ();
void randomize();
void warmup_random (double nsga_seed);


int nsga_popsize=0;
double nsga_seed;
double nsga_oldrand[55];
int nsga_jrand;
int nsga_nobj=NSGA2_NOBJ;

/* Definition of random number generation routines */


/* Get nsga_seed number for random and start it up */
void randomize() {
    int j1;
    for(j1=0; j1<=54; j1++) {
        nsga_oldrand[j1] = 0.0;
    }
    nsga_jrand=0;
    warmup_random (nsga_seed);
    return;
}

/* Get randomize off and running */
void warmup_random (double nsga_seed) {
    int j1, ii;
    double new_random, prev_random;
    nsga_oldrand[54] = nsga_seed;
    new_random = 0.000000001;
    prev_random = nsga_seed;
    for(j1=1; j1<=54; j1++) {
        ii = (21*j1)%54;
        nsga_oldrand[ii] = new_random;
        new_random = prev_random-new_random;
        if(new_random<0.0) {
            new_random += 1.0;
        }
        prev_random = nsga_oldrand[ii];
    }
    advance_random ();
    advance_random ();
    advance_random ();
    nsga_jrand = 0;
    return;
}

/* Create next batch of 55 random numbers */
void advance_random () {
    int j1;
    double new_random;
    for(j1=0; j1<24; j1++) {
        new_random = nsga_oldrand[j1]-nsga_oldrand[j1+31];
        if(new_random<0.0) {
            new_random = new_random+1.0;
        }
        nsga_oldrand[j1] = new_random;
    }
    for(j1=24; j1<55; j1++) {
        new_random = nsga_oldrand[j1]-nsga_oldrand[j1-24];
        if(new_random<0.0) {
            new_random = new_random+1.0;
        }
        nsga_oldrand[j1] = new_random;
    }
}

/* Fetch a single random number between 0.0 and 1.0 */
double randomperc() {
    nsga_jrand++;
    if(nsga_jrand>=55) {
        nsga_jrand = 1;
        advance_random();
    }
    return((double)nsga_oldrand[nsga_jrand]);
}

/* Fetch a single random integer between low and high including the bounds */
int rnd (int low, int high) {
    int res;
    if (low >= high) {
        res = low;
    } else {
        res = low + (randomperc()*(high-low+1));
        if (res > high) {
            res = high;
        }
    }
    return (res);
}

/* Fetch a single random real number between low and high including the bounds */
double rndreal (double low, double high) {
    return (low + (high-low)*randomperc());
}


/* Function to allocate memory to a NSGA_population */
void allocate_memory_pop (NSGA_population *pop, int size) {
    int i;
    pop->ind = (individual *)malloc(size*sizeof(individual));
    for (i=0; i<size; i++) {
        allocate_memory_ind (&(pop->ind[i]));
    }
    return;
}

/* Function to allocate memory to an individual */
void allocate_memory_ind (individual *ind) {
    ind->obj = (double *)malloc(nsga_nobj*sizeof(double));
    return;
}

/* Function to deallocate memory to a NSGA_population */
void deallocate_memory_pop (NSGA_population *pop, int size) {
    int i;
    for (i=0; i<size; i++) {
        deallocate_memory_ind (&(pop->ind[i]));
    }
    free (pop->ind);
    return;
}

/* Function to deallocate memory to an individual */
void deallocate_memory_ind (individual *ind) {
    free(ind->obj);
    return;
}
/* Crowding distance computation routines */


/* Routine to compute crowding distance based on ojbective function values when the NSGA_population in in the form of a listNSGA */
void assign_crowding_distance_listNSGA (NSGA_population *pop, listNSGA *lst,
                                        int front_size) {
    int **obj_array;
    int *dist;
    int i, j;
    listNSGA *temp;
    temp = lst;
    if (front_size==1) {
        pop->ind[lst->index].crowd_dist = INF;
        return;
    }
    if (front_size==2) {
        pop->ind[lst->index].crowd_dist = INF;
        pop->ind[lst->child->index].crowd_dist = INF;
        return;
    }
    obj_array = (int **)malloc(nsga_nobj*sizeof(int*));
    dist = (int *)malloc(front_size*sizeof(int));
    for (i=0; i<nsga_nobj; i++) {
        obj_array[i] = (int *)malloc(front_size*sizeof(int));
    }
    for (j=0; j<front_size; j++) {
        dist[j] = temp->index;
        temp = temp->child;
    }
    assign_crowding_distance (pop, dist, obj_array, front_size);
    free (dist);
    for (i=0; i<nsga_nobj; i++) {
        free (obj_array[i]);
    }
    free (obj_array);
    return;
}

/* Routine to compute crowding distance based on objective function values when the NSGA_population in in the form of an array */
void assign_crowding_distance_indices (NSGA_population *pop, int c1, int c2) {
    int **obj_array;
    int *dist;
    int i, j;
    int front_size;
    front_size = c2-c1+1;
    if (front_size==1) {
        pop->ind[c1].crowd_dist = INF;
        return;
    }
    if (front_size==2) {
        pop->ind[c1].crowd_dist = INF;
        pop->ind[c2].crowd_dist = INF;
        return;
    }
    obj_array = (int **)malloc(nsga_nobj*sizeof(int*));
    dist = (int *)malloc(front_size*sizeof(int));
    for (i=0; i<nsga_nobj; i++) {
        obj_array[i] = (int *)malloc(front_size*sizeof(int));
    }
    for (j=0; j<front_size; j++) {
        dist[j] = c1++;
    }
    assign_crowding_distance (pop, dist, obj_array, front_size);
    free (dist);
    for (i=0; i<nsga_nobj; i++) {
        free (obj_array[i]);
    }
    free (obj_array);
    return;
}

/* Routine to compute crowding distances */
void assign_crowding_distance (NSGA_population *pop, int *dist, int **obj_array,
                               int front_size) {
    int i, j;
    for (i=0; i<nsga_nobj; i++) {
        for (j=0; j<front_size; j++) {
            obj_array[i][j] = dist[j];
        }
        quicksort_front_obj (pop, i, obj_array[i], front_size);
    }
    for (j=0; j<front_size; j++) {
        pop->ind[dist[j]].crowd_dist = 0.0;
    }
    for (i=0; i<nsga_nobj; i++) {
        pop->ind[obj_array[i][0]].crowd_dist = INF;
    }
    for (i=0; i<nsga_nobj; i++) {
        for (j=1; j<front_size-1; j++) {
            if (pop->ind[obj_array[i][j]].crowd_dist != INF) {
                if (pop->ind[obj_array[i][front_size-1]].obj[i] ==
                        pop->ind[obj_array[i][0]].obj[i]) {
                    pop->ind[obj_array[i][j]].crowd_dist += 0.0;
                } else {
                    pop->ind[obj_array[i][j]].crowd_dist += (pop->ind[obj_array[i][j+1]].obj[i] -
                                                            pop->ind[obj_array[i][j-1]].obj[i])/(pop->ind[obj_array[i][front_size-1]].obj[i]
                                                                    - pop->ind[obj_array[i][0]].obj[i]);
                }
            }
        }
    }
    for (j=0; j<front_size; j++) {
        if (pop->ind[dist[j]].crowd_dist != INF) {
            pop->ind[dist[j]].crowd_dist = (pop->ind[dist[j]].crowd_dist)/nsga_nobj;
        }
    }
    return;
}
/* Domination checking routines */


/* Routine for usual non-domination checking
   It will return the following values
   1 if a dominates b
   -1 if b dominates a
   0 if both a and b are non-dominated */

int check_dominance (individual *a, individual *b) {
    int i;
    int flag1;
    int flag2;
    flag1 = 0;
    flag2 = 0;
    for (i=0; i<nsga_nobj; i++) {
        if (a->obj[i] < b->obj[i]) {
            flag1 = 1;

        } else {
            if (a->obj[i] > b->obj[i]) {
                flag2 = 1;
            }
        }
    }
    if (flag1==1 && flag2==0) {
        return (1);
    } else {
        if (flag1==0 && flag2==1) {
            return (-1);
        } else {
            return (0);
        }
    }
}
/* A custom doubly linked listNSGA implemenation */


/* Insert an element X into the listNSGA at location specified by NODE */
void insert (listNSGA *node, int x) {
    listNSGA *temp;
    if (node==NULL) {
        printf("\n Error!! asked to enter after a NULL pointer, hence exiting \n");
        exit(1);
    }
    temp = (listNSGA *)malloc(sizeof(listNSGA));
    temp->index = x;
    temp->child = node->child;
    temp->parent = node;
    if (node->child != NULL) {
        node->child->parent = temp;
    }
    node->child = temp;
    return;
}

/* Delete the node NODE from the listNSGA */
listNSGA* del (listNSGA *node) {
    listNSGA *temp;
    if (node==NULL) {
        printf("\n Error!! asked to delete a NULL pointer, hence exiting \n");
        exit(1);
    }
    temp = node->parent;
    temp->child = node->child;
    if (temp->child!=NULL) {
        temp->child->parent = temp;
    }
    free (node);
    return (temp);
}
/* Routines for randomized recursive quick-sort */


/* Randomized quick sort routine to sort a NSGA_population based on a particular objective chosen */
void quicksort_front_obj(NSGA_population *pop, int objcount, int obj_array[],
                         int obj_array_size) {
    q_sort_front_obj (pop, objcount, obj_array, 0, obj_array_size-1);
    return;
}

/* Actual implementation of the randomized quick sort used to sort a NSGA_population based on a particular objective chosen */
void q_sort_front_obj(NSGA_population *pop, int objcount, int obj_array[],
                      int left, int right) {
    int index;
    int temp;
    int i, j;
    double pivot;
    if (left<right) {
        index = rnd (left, right);
        temp = obj_array[right];
        obj_array[right] = obj_array[index];
        obj_array[index] = temp;
        pivot = pop->ind[obj_array[right]].obj[objcount];
        i = left-1;
        for (j=left; j<right; j++) {
            if (pop->ind[obj_array[j]].obj[objcount] <= pivot) {
                i+=1;
                temp = obj_array[j];
                obj_array[j] = obj_array[i];
                obj_array[i] = temp;
            }
        }
        index=i+1;
        temp = obj_array[index];
        obj_array[index] = obj_array[right];
        obj_array[right] = temp;
        q_sort_front_obj (pop, objcount, obj_array, left, index-1);
        q_sort_front_obj (pop, objcount, obj_array, index+1, right);
    }
    return;
}

/* Randomized quick sort routine to sort a NSGA_population based on crowding distance */
void quicksort_dist(NSGA_population *pop, int *dist, int front_size) {
    q_sort_dist (pop, dist, 0, front_size-1);
    return;
}

/* Actual implementation of the randomized quick sort used to sort a NSGA_population based on crowding distance */
void q_sort_dist(NSGA_population *pop, int *dist, int left, int right) {
    int index;
    int temp;
    int i, j;
    double pivot;
    if (left<right) {
        index = rnd (left, right);
        temp = dist[right];
        dist[right] = dist[index];
        dist[index] = temp;
        pivot = pop->ind[dist[right]].crowd_dist;
        i = left-1;
        for (j=left; j<right; j++) {
            if (pop->ind[dist[j]].crowd_dist <= pivot) {
                i+=1;
                temp = dist[j];
                dist[j] = dist[i];
                dist[i] = temp;
            }
        }
        index=i+1;
        temp = dist[index];
        dist[index] = dist[right];
        dist[right] = temp;
        q_sort_dist (pop, dist, left, index-1);
        q_sort_dist (pop, dist, index+1, right);
    }
    return;
}
/* Rank assignment routine */


/* Function to assign rank and crowding distance to a NSGA_population of size pop_size*/
void assign_rank_and_crowding_distance (NSGA_population *new_pop) {
    int flag;
    int i;
    int end;
    int front_size;
    int rank=1;
    listNSGA *orig;
    listNSGA *cur;
    listNSGA *temp1, *temp2;
    orig = (listNSGA *)malloc(sizeof(listNSGA));
    cur = (listNSGA *)malloc(sizeof(listNSGA));
    front_size = 0;
    orig->index = -1;
    orig->parent = NULL;
    orig->child = NULL;
    cur->index = -1;
    cur->parent = NULL;
    cur->child = NULL;
    temp1 = orig;
    for (i=0; i<nsga_popsize; i++) {
        insert (temp1,i);
        temp1 = temp1->child;
    }
    do {
        if (orig->child->child == NULL) {
            new_pop->ind[orig->child->index].rank = rank;
            new_pop->ind[orig->child->index].crowd_dist = INF;
            break;
        }
        temp1 = orig->child;
        insert (cur, temp1->index);
        front_size = 1;
        temp2 = cur->child;
        temp1 = del (temp1);
        temp1 = temp1->child;
        do {
            temp2 = cur->child;
            do {
                end = 0;
                flag = check_dominance (&(new_pop->ind[temp1->index]),
                                        &(new_pop->ind[temp2->index]));
                if (flag == 1) {
                    insert (orig, temp2->index);
                    temp2 = del (temp2);
                    front_size--;
                    temp2 = temp2->child;
                }
                if (flag == 0) {
                    temp2 = temp2->child;
                }
                if (flag == -1) {
                    end = 1;
                }
            } while (end!=1 && temp2!=NULL);
            if (flag == 0 || flag == 1) {
                insert (cur, temp1->index);
                front_size++;
                temp1 = del (temp1);
            }
            temp1 = temp1->child;
        } while (temp1 != NULL);
        temp2 = cur->child;
        do {
            new_pop->ind[temp2->index].rank = rank;
            temp2 = temp2->child;
        } while (temp2 != NULL);
        assign_crowding_distance_listNSGA (new_pop, cur->child, front_size);
        temp2 = cur->child;
        do {
            temp2 = del (temp2);
            temp2 = temp2->child;
        } while (cur->child !=NULL);
        rank+=1;
    } while (orig->child!=NULL);
    free (orig);
    free (cur);
    return;
}

int comparator(const void *p, const void *q) {

    if (((individual *) p)->rank != ((individual *) q)->rank)
        return (((individual *) p)->rank >  ((individual *) q)->rank);
    else
        return ( (individual *) p )->crowd_dist > ((individual *) q)->crowd_dist;

}




void select(CIndividual* children, size_t numChildren,
            CIndividual* parents, size_t numParents,
            bool mixedpop) {

    int i;
    int j;
    NSGA_population *nsga_pop;


    if (mixedpop)
        nsga_popsize = numChildren + numParents;
    else
        nsga_popsize = numChildren;


    nsga_pop = (NSGA_population *)malloc(sizeof(NSGA_population));
    allocate_memory_pop (nsga_pop,nsga_popsize);

    randomize(); /*FIXME: check how to deal with EASEA own random  routines*/




    if (mixedpop) {
        for (i = 0; i < (int) numParents; i++) {
            nsga_pop->ind[i].id=i;
            for(j=0; j<nsga_nobj; j++) {
                nsga_pop->ind[i].obj[j] = parents[i].fitness;
            }

        }

        for (i = 0; i < (int) numChildren; i++) {
            nsga_pop->ind[numParents + i].id = i + numParents;
            for(j=0; j<nsga_nobj; j++) {
                nsga_pop->ind[numParents + i].obj[j] =
                    children[i].fitness;
            }

        }

    } else {
        for (i=0; i<(int)nsga_popsize; i++) {
            nsga_pop->ind[i].id=i;
            for(j=0; j<nsga_nobj; j++) {
                nsga_pop->ind[i].obj[j] = parents[i].fitness;
            }

        }
    }

    assign_rank_and_crowding_distance (nsga_pop);
    qsort((void *)nsga_pop->ind,nsga_popsize,sizeof(nsga_pop->ind[0]),comparator);


    for (i=0; i<nsga_popsize; i++) {
        /*
        		printf("[%d]->id %d\n",i,nsga_pop->ind[i].id);
        		printf("[%d]->crowd_dist %f\n",i,nsga_pop->ind[i].crowd_dist);
        		printf("[%d]->rank %d\n",i,nsga_pop->ind[i].rank);
        		printf("[%d]->f0 %d %f ->f1 %f\n",i,nsga_pop->ind[i].obj[0], nsga_pop->ind[i].obj[1]);
        */
        if (mixedpop) {
            if (nsga_pop->ind[i].id < numParents) {
                CIndividual* indiv = &(parents[nsga_pop->ind[i].id]);
                indiv->rank = nsga_pop->ind[i].rank;
            } else {
                CIndividual* indiv = &(children[nsga_pop->ind[i].id - numParents]);
                indiv->rank = nsga_pop->ind[i].rank;
            }
        } else {
            CIndividual* indiv = &(parents[nsga_pop->ind[i].id]);
            indiv->rank = nsga_pop->ind[i].rank;
        }
    }

    deallocate_memory_pop(nsga_pop,nsga_popsize);
}
}
