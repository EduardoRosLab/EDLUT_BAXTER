/***************************************************************************
 *                           ExponentialTable.h                            *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <cmath>
#include <iostream>

#include "../../include/simulation/ExponentialTable.h"


const float ExponentialTable::Min=-20.0f;

const float ExponentialTable::Max=20.0f;

const float ExponentialTable::aux=(ExponentialTable::TableSize-1)/( ExponentialTable::Max- ExponentialTable::Min);
   		



float * ExponentialTable::LookUpTable = generate_data();


ExponentialTable::~ExponentialTable(void){
	if(this->LookUpTable!=0){
		delete this->LookUpTable;
		this->LookUpTable=0;
	}
}