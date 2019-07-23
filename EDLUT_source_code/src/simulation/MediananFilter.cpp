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

#include <cmath>
#include <iostream>

#include "../../include/simulation/MediananFilter.h"



MediananFilter::MediananFilter(){
	datpoint=buffer;  /* pointer into circular buffer of data */
	small_point.point=NULL;
	small_point.value=STOPPER;  /* chain stopper. */
	big_point.point=&small_point;
	big_point.value=0;  /* pointer to head (largest) of linked list.*/

	index=0;
}


MediananFilter::~MediananFilter(void){
}


double MediananFilter::mediananfilter(double datum){
	int i;

    if( (++datpoint - buffer) >= NWIDTH) datpoint=buffer;  /* increment and wrap data in pointer.*/
    datpoint->value=datum        ;  /* Copy in new datum */
    successor=datpoint->point    ;  /* save pointer to old value's successor */
    median = &big_point                ;  /* median initially to first in chain */
    scanold = NULL               ;  /* scanold initially null. */
    scan = &big_point                  ;  /* points to pointer to first (largest) datum in chain */
  /* Handle chain-out of first item in chain as special case */
        if( scan->point == datpoint ) scan->point = successor;
        scanold = scan ;            /* Save this pointer and   */
        scan = scan->point ;        /* step down chain */
  /* loop through the chain, normal loop exit via break. */
    for( i=0 ;i<NWIDTH ; i++ )
    {
     /* Handle odd-numbered item in chain  */
        if( scan->point == datpoint ) scan->point = successor;  /* Chain out the old datum.*/
        if( (scan->value < datum) )        /* If datum is larger than scanned value,*/
        {
            datpoint->point = scanold->point;          /* chain it in here.  */
            scanold->point = datpoint;          /* mark it chained in. */
            datum = STOPPER;
        };
  /* Step median pointer down chain after doing odd-numbered element */
        median = median->point       ;       /* Step median pointer.  */
        if ( scan == &small_point ) break ;        /* Break at end of chain  */
        scanold = scan ;          /* Save this pointer and   */
        scan = scan->point ;            /* step down chain */
  /* Handle even-numbered item in chain.  */
        if( scan->point == datpoint ) scan->point = successor; 
        if( (scan->value < datum) )         
        {
            datpoint->point = scanold->point;       
            scanold->point = datpoint;
            datum = STOPPER;
        };
        if ( scan == &small_point ) break;
        scanold = scan ;                            
        scan = scan->point;
    };
    return( median->value );
}



//double MediananFilter::medianfilter(double datum){
//	valuesOfMedianFilter[index]=datum;
//	index++;
//	index=index%NWIDTH;
//
//	double value=0;
//	for(int i=0; i<NWIDTH; i++){
//		value+=valuesOfMedianFilter[i];
//	}
//	value/=NWIDTH;
//	return value;
//}

double MediananFilter::medianfilter(double datum){
	valuesOfMedianFilter[index]=datum;

	int aux_index=index;

	double value=0;
	for(int j=NWIDTH; j>0; j--){

		value+=valuesOfMedianFilter[aux_index]*j;
		if(aux_index==0){
			aux_index=NWIDTH;
		}
		aux_index--;
		
	}
	value/=(NWIDTH*(NWIDTH+1.0))/2.0;

	index++;
	index=index%NWIDTH;

	return value;
}