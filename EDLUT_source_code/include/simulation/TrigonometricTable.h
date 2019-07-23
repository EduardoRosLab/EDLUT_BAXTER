/***************************************************************************
 *                           TrigonometricTable.h                          *
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

#ifndef TRIGONOMETRICTABLE_H_
#define TRIGONOMETRICTABLE_H_

#include <cmath>

/*!
 * \file TrigonometricTable.h
 *
 * \author Francisco Naveros
 * \date February 2015
 *
 * This file declares a look-up table for a sinusoidal and cosenoidal function.
 */
 


class TrigonometricTable{
	
   	public:
   		
		/*!
		 * Precalculated sin and cos terms.
		 */
		static float * TrigonometricLUT;

		/*!
		 * Precalculated LUT Step.
		 */
		static const float LUTStep;
		static const float inv_LUTStep;

		/*!
		 * Number of elements stored in TrigonometricLUT for sin and cos function.
		 */
		static const int N_ELEMENTS=1024*1024;

		~TrigonometricTable();

		/*!
		 * \brief It calculates the trigonometric functions.
		 *
		 * It calculates the trigonometric functions.
		 *
		 * \return The trigonometric functions values.
		 */
		static float * GenerateTrigonometricLUT(){
			float * NewSinLUT=new float[N_ELEMENTS*2];

			for (unsigned int i=0; i<N_ELEMENTS; ++i){
				NewSinLUT[2*i] = sinf(LUTStep*i);
				NewSinLUT[2*i+1] = cosf(LUTStep*i);
			}

			return NewSinLUT;
		}


		/*!
		 * \brief It computes the index inside the table for a value.
		 *
		 * It computes the index inside the table for a value.
		 *
		 * \param ElapsedRelative value over the trigonometric function must be calculated
		 *
		 * \return The index inside the table for a value.
		 */
		static int CalculateOffsetPosition(float ElapsedRelative){
			return ((int)(ElapsedRelative*inv_LUTStep + 0.5f))*2;
		}

		/*!
		 * \brief It compute a new index inside the table.
		 *
		 * It compute a new index inside the table.
		 *
		 * \param initPosition Initial index inside the table.
		 * \param offset Index increment.
		 *
		 * \return The index inside the table.
		 */
		static float CalculateValidPosition(int initPosition, int offset){
			return((initPosition+offset)%(N_ELEMENTS*2));
		}


		/*!
		 * \brief It gets a trigonometric value.
		 *
		 * It gets a trigonometric value.
		 *
		 * \param index
		 *
		 * \return The trigonometric value.
		 */
		static float GetElement(int index){
			return TrigonometricLUT[index];
		}
  	
};


#endif /*EXPONENTIALTABLE_H_*/



