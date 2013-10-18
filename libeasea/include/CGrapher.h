/*
 *    Copyright (C) 2009  Ogier Maitre

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

#ifndef CGRAPHER_H_
#define CGRAPHER_H_

class CRandomGenerator;
#include <iostream>
#include <stdlib.h>
#include "Parameters.h"

class CGrapher {
public:
	FILE *fWrit;
	FILE *fRead;
	int pid;
	int valid;
public:
	CGrapher(Parameters* param, char* title);
	~CGrapher();
};

#endif /* CGRAPHER_H_ */
