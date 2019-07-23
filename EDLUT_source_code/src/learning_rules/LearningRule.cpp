/***************************************************************************
 *                           LearningRule.cpp                              *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido                        *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/learning_rules/LearningRule.h"


LearningRule::LearningRule(int NewLearningRuleIndex): State(0), counter(0), LearningRuleIndex (NewLearningRuleIndex){

}

LearningRule::~LearningRule(){
	if(State!=0){
		delete State;
	}
}



ConnectionState * LearningRule::GetConnectionState(){
	return this->State;
}



