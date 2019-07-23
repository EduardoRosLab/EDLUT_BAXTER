/***************************************************************************
 *                           TrigonometricTable.cpp                        *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros                    *
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

#include "../../include/simulation/TrigonometricTable.h"


const float TrigonometricTable::LUTStep = 2.0f*4.0f*atan(1.0f)/TrigonometricTable::N_ELEMENTS;
const float TrigonometricTable::inv_LUTStep = 1.0f/TrigonometricTable::LUTStep;

float * TrigonometricTable::TrigonometricLUT=GenerateTrigonometricLUT();


TrigonometricTable::~TrigonometricTable(void){
	if(this->TrigonometricLUT!=0){
		delete this->TrigonometricLUT;
		this->TrigonometricLUT=0;
	}
}