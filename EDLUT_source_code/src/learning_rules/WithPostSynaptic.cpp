/***************************************************************************
 *                           WithPostSynaptic.cpp                          *
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

#include "../../include/learning_rules/WithPostSynaptic.h"


WithPostSynaptic::WithPostSynaptic(int NewLearningRuleIndex):LearningRule(NewLearningRuleIndex){

}

WithPostSynaptic::~WithPostSynaptic(){

}

bool WithPostSynaptic::ImplementPostSynaptic(){
	return true;
}



