/*
 * CIndividual.h
 *
 *  Created on: 23 juin 2009
 *      Author: maitre
 */

#ifndef CINDIVIDUAL_H_
#define CINDIVIDUAL_H_

class CRandomGenerator;
#include <iostream>
#include <string>

using namespace std;

class CIndividual {
public:
	bool valid;  // Est il valide (Contraintes sur les individus respectées)?
	bool isImmigrant;  // Vient il d'un autre ilot?
	float fitness;  // La fonction de fitness de l'individu
	/*----------------------------------------------------*/
	int ident; // un entier qui donne l'identité de l'individu
	char origine; // vaut A si Ancètre, M si mutation, C si crossover R si copie B si both
	int parent1;
	int parent2;
	int survival;
	float gainFitness;
	/*----------------------------------------------------*/
	static CRandomGenerator* rg; // Pour l'utiliser lors des mutations ou les cross-overs 
public:
	CIndividual();  // Constructeur
	//CIndividual(const CIndividual& indiv);
	
	/////Fonctions virtuelles car les fonctions sont implémentées lors de la compilation du .ez
	virtual ~CIndividual();  // Destructeur
	virtual float evaluate() = 0; //Evaluation 
	virtual void printOn(std::ostream& O) const = 0; // Impression
	virtual unsigned mutate(float pMutationPerGene) = 0; //Mutation
	virtual CIndividual* crossover(CIndividual** p2) = 0; //Crossover
	virtual CIndividual* clone() = 0; //Clonage

        virtual std::string serialize() = 0; //transforme en chaine de caractère pour envoyer l'individu dans un socket
        virtual void deserialize(std::string EASEA_Line) = 0;

	virtual void boundChecking() = 0; //Regarder si les contraintes sont respectées

	static unsigned getCrossoverArrity(){ return 2; }
	float getFitness(){ return this->fitness; }


};

#endif /* CINDIVIDUAL_H_ */
