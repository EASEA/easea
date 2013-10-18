/*
 * CIndividual.h

     Copyright (C) 2009  Ogier Maitre

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
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
