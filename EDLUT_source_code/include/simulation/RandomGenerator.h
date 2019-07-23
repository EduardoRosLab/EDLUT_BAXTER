/***************************************************************************
 *                           RandomGenerator.h	                           *
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

#ifndef RANDOMGENERATOR_H_
#define RANDOMGENERATOR_H_

#include <cmath>

/*!
 * \file RandomGenerator.h
 *
 * \author Francisco Naveros
 * \date April 2015
 *
 * This file declares a random generator. We use this one in order to obtain the same simulation result in Windows, Unix and Mac OS.
 */
 


class RandomGenerator{
	
   	public:
   		
	/*!
	* Maximum rand that can be generated
	*/
	static const int MAX_RAND=2147483647;//(2^31)-1 

	/*!
	* Auxiliar value used to calculate the next value based in a previous one.
	*/
	static unsigned long int next_element;  	

	/*!
	 * \brief It sets the rand generator seed. 
	 *
	 * It sets the rand generator seed. 
	 *
	 * \param seed Rand generator seed.
	 */
	static void srand(unsigned int seed){
		next_element = seed;
	}

	/*!
	 * \brief It gets an integer rand value.
	 *
	 * It gets an integer rand value between 0 and MAX_RAND.
	 *
	 * \return An integer rand value.
	 */
	static unsigned int rand(void){
		next_element = ((next_element * 1103515245) + 12345) & 0x7fffffff;
		return (unsigned int)next_element;
	}
	
	/*!
	 * \brief It gets a float rand value.
	 *
	 * It gets a float rand value between 0 and 1.
	 *
	 * \return A float rand value.
	 */
	static float frand(void){
		next_element = ((next_element * 1103515245) + 12345) & 0x7fffffff;
		return ((float)next_element)/MAX_RAND;
	}

	/*!
	 * \brief It gets a double rand value.
	 *
	 * It gets a double rand value between 0 and 1.
	 *
	 * \return A double rand value.
	 */
	static double drand(void){
		next_element = ((next_element * 1103515245) + 12345) & 0x7fffffff;
		return ((double)next_element)/MAX_RAND;
	}

};



#endif /*RANDOMGENERATOR_H_*/



