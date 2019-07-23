/***************************************************************************
 *                           MediananFilter.cpp                            *
 *                           -------------------                           *
 * copyright            : (C) 2016 by Francisco Naveros and Niceto Luque   *
 * email                : fnaveros@ugr.es and nluque@ugr.es                *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef MEDIANANFILTER_H_
#define MEDIANANFILTER_H_



/*!
 * \file MediananFilter.h
 *
 * \author Francisco Naveros
 * \date April 2015
 *
 * This file declares a class which implements a medianan filter of size N.
 */

#include <stdio.h>
#define NWIDTH 5
#define STOPPER 0.0 /* Smaller than any datum */


struct valuesOfMediananFilter
{
    struct valuesOfMediananFilter   *point;  /* Pointers forming list linked in sorted order */
    double  value;  /* Values to sort */
};
 


class MediananFilter{
	
   	public:

		valuesOfMediananFilter buffer[NWIDTH];  /* Buffer of nwidth pairs */
		valuesOfMediananFilter *datpoint;  /* pointer into circular buffer of data */;
		valuesOfMediananFilter small_point;  /* chain stopper. */
		valuesOfMediananFilter big_point;  /* pointer to head (largest) of linked list.*/
		valuesOfMediananFilter *successor   ;  /* pointer to successor of replaced data item */
		valuesOfMediananFilter *scan        ;  /* pointer used to scan down the sorted list */
		valuesOfMediananFilter *scanold     ;  /* previous value of scan */
		valuesOfMediananFilter *median;     ;  /* pointer to median */
   		
		 /*!
   		 * Filter size
   		 */
		double valuesOfMedianFilter[NWIDTH];
		int index;


		MediananFilter();

		~MediananFilter();

		double mediananfilter(double datum);

		double medianfilter(double datum);


  	
};


#endif /*MEDIANANFILTER_H_*/



