/***************************************************************************
 *                           NeuronModelTable.h                            *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
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

#ifndef NEURONMODELTABLE_H_
#define NEURONMODELTABLE_H_

/*!
 * \file NeuronModelTable.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a neuron model table.
 */

//#define table_indcomp(dim,coo) (((coo)>(dim)->vlast)?((dim)->size-1):(((coo)<(dim)->vfirst)?0:(*((dim)->vindex+(int)( ((coo) - (dim)->vfirst) * (dim)->inv_vscale + 0.49) ))))
//#define table_indcomp2(dim,coo) (((coo)>(dim)->vlast)?((dim)->size-1):(((coo)<(dim)->vfirst)?0:(*((dim)->vindex+(int)( ((coo) - (dim)->vfirst) * (dim)->inv_vscale + 0.5 + *((dim)->voffset+(int)( ((coo) - (dim)->vfirst) * (dim)->inv_vscale + 0.5) ) ) ))))
//#define table_ind_int(dim,coo) (*((dim)->vindex+(int)( ((coo) - (dim)->vfirst) * (dim)->inv_vscale + (*((dim)->voffset+(int)( ((coo) - (dim)->vfirst) * (dim)->inv_vscale) ))) ))


#include <cstdio>
#include <cstdlib>
#include <cmath>
#if (defined (_WIN32) || defined(_WIN64))
	#include "../stdint_WIN.h"
#else 
	#include <stdint.h>
#endif



#include "../spike/EDLUTFileException.h"

#include "../simulation/Configuration.h"

class VectorNeuronState;

/*!
 * \class NeuronModelTable
 *
 * \brief Precalculated table of a neuron model.
 *
 * This class abstract the behaviour of a neuron model table. These tables are
 * used for access to the behaviour of a neuron model. These tables are loaded
 * from a file and externally generated.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class NeuronModelTable {
	
	public:
   	
   		/*!
		 * \class TableDimension
		 *
		 * \brief Dimension of a neuron model table.
		 *
		 * This class abstract the behaviour of a neuron model table dimension. Each table
		 * is composed of different dimensions (time, membrane potential...).
		 *
		 * \author Jesus Garrido
		 * \author Richard Carrillo
		 * \date August 2008
		 */
   		class TableDimension {
   			
   			public:
   			
   				/*!
   				 * Size of the table dimension. Number of elements in the dimension.
   				 */
   				unsigned long size;
   			
   				/*!
   				 * Values of the dimension.
   				 */
   				float *coord;
   			
   				/*!
   				 * Virtual index of the dimension
   				 */
   				int *vindex;
   			
   				/*!
   				 * Virtual offsets of the dimension.
   				 */
   				float *voffset;

   				/*!
   				 * Virtual index and offsets of the dimension
   				 */
				float *vindex_voffset;
							
   				/*! 
   				 * Virtual scale of the dimension.
   				 */
   				float vscale;
				float inv_vscale;
   				
   				/*!
   				 * First element of the dimension.
   				 */
   				float vfirst;

				/*!
   				 * Last element of the dimension.
   				 */
				float vlast;

   				/*!
   				 * State variable of the dimension.
   				 */
   				int statevar;
   			
   				/*!
   				 * Interpolation method of the dimension.
   				 * 0: table_access_direct (interpolation is not used)
   				 * 1: interp bilinear
   				 * 2: interp linear
   				 * 3: interp linear_ex
   				 * 4: interp linear from 2 different positions
   				 * 5: interp linear form n different positions
   				 */
   				int interp;
   			
   				/*!
   				 * Next interpolated dimension.
   				 */
   				int nextintdim;
   				
   				/*!
   				 * \brief Default constructor.
   				 * 
   				 * It creates and initializes a new table dimension with the default values.
   				 */
   				TableDimension();
   				
   				/*!
   				 * \brief Class destructor.
   				 * 
   				 * It destroys an object and frees the used memory.
   				 */
   				~TableDimension();

		
				int table_indcomp(float coo);
				int table_indcomp2(float coo);
				int table_ind_int(float coo);
				float check_range(float value);
  		};
  		
  		/*!
  		 * \brief Default constructor.
  		 * 
  		 * It creates and initializes a new neuron model table with their default values.
  		 */
  		NeuronModelTable();
  		
  		/*!
  		 * \brief Class destructor.
  		 * 
  		 * It destroys an object and frees  the used memory.
  		 */
  		~NeuronModelTable();
  		
  		/*!
  		 * \brief It prints information about the load table.
  		 * 
  		 * It prints information about the load table.
  		 * 
  		 */
  		void TableInfo();
  		
  		/*!
  		 * \brief It loads a neuron model table.
  		 * 
  		 * It loads the neuron model table values from the .dat file.
  		 * 
  		 * \param fd Neuron model table file descriptor. 
  		 * 
  		 * \pre The file descriptor must be where this table starts, so this is thought for loading all tables in order.
  		 * \post The file descriptor is where this table description ends.
  		 * 
  		 * \throw EDLUTException If something wrong happens.
  		 */  		
  		void LoadTable(FILE *fd) throw (EDLUTException);
  		
  		/*!
  		 * \brief It loads the table description from a file.
  		 * 
  		 * It loads the table description from a .cfg file.
  		 * 
		 * \param ConfigFile The file name 
  		 * \param fh The file descriptor of the neuron model table description.
  		 * \param Currentline The line in the file where we are reading (for error description reasons).
  		 * \pre The file descriptor must be where this table description starts, so this is thought for loading all tables in order.
  		 * \post The file descriptor is where this table description ends.
  		 * \post Currentline is modified with the next line in the file.
  		 * 
  		 */
		void LoadTableDescription(string ConfigFile, FILE *fh, long & Currentline) throw (EDLUTFileException);
  		
  		//void TableInfo();
  		
  		/*!
  		 * \brief It gets a table dimension.
  		 * 
  		 * It gets a table dimension of the neuron model table.
  		 * 
  		 * \param index The index of the dimension.
  		 * 
  		 * \return The indexth table dimension.
  		 */
  		const TableDimension * GetDimensionAt(int index) const;
  		
  		/*!
  		 * \brief It sets a table dimension.
  		 * 
  		 * It sets a table dimension of the neuron model table.
  		 * 
  		 * \param index The index of the dimension to set.
  		 * \param Dimension The new dimension of the neuron model table.
  		 */
  		void SetDimensionAt(int index, TableDimension Dimension);
  		
  		/*!
  		 * \brief It gets the number of dimensions.
  		 * 
  		 * It gets the number of table dimensions.
  		 * 
  		 * \return The number of table dimensions.
  		 */
  		unsigned long GetDimensionNumber() const;
  		
  		/*!
  		 * \brief It gets an element of the table.
  		 * 
  		 * It gets an concret element of the table.
  		 * 
  		 * \param index The index of the element to get.
  		 * 
  		 * \return The value of the indexth element.
  		 */
  		float GetElementAt(int index) const;
  		
  		/*!
  		 * \brief It sets an element of the table.
  		 * 
  		 * It sets an concret element of the table.
  		 * 
  		 * \param index The index of the element to set.
  		 * \param Element The new value of the element.
  		 */
  		void SetElementAt(int index, float Element);
  		
  		/*!
  		 * \brief It gets the number of elements.
  		 * 
  		 * It gets the number of elements in the table.
  		 * 
  		 * \return The number of elements in the table.
  		 */
  		unsigned long GetElementsNumber() const;
  		
  		/*!
  		 * \brief It gets the interpolation of the dimensions.
  		 * 
  		 * It gets the interpolation of the dimensions.
  		 * 
  		 * \return The interpolation of the dimensions.
  		 */
  		int GetInterpolation() const;
  		
  		/*!
  		 * \brief It gets the first interpolated dimension.
  		 * 
  		 * It gets the first interpolated dimension.
  		 * 
  		 * \return The first interpolated dimension.
  		 */
  		int GetFirstInterpolation() const;
  		
  		/*!
  		 * \brief It gets the table value.
  		 * 
  		 * It gets the table value with concrete state variables and the 
  		 * current interpolation method.
  		 * 
  		 * \param statevars The current state variables of the neuron.
  		 * 
  		 * \return The value of the table (with or without interpolation).
  		 */
  		float TableAccess(int index, VectorNeuronState * statevars);

		/*!
   		 * \brief It gets a table value without interpolation.
   		 * 
   		 * It gets a table value without interpolation with concrete state variables.
   		 * 
   		 * \param statevars State variables of the neuron.
   		 * 
   		 * \return The table value with that state.
   		 */
   		float TableAccessDirect(int index, VectorNeuronState * statevars);

		/*!
   		 * \brief It sets the state variable index.
   		 * 
   		 * It sets the state variable index.
   		 * 
   		 * \param newstatevariableindex state variable index.
   		 */
		void SetOutputStateVariableIndex(int newoutputstatevariableindex);

		 /*!
   		 * \brief It gets the state variable index.
   		 * 
   		 * It gets the state variable index.
   		 * 
   		 * \return The state variable index.
   		 */
		int GetOutputStateVariableIndex();

		/*!
   		 * \brief It calculates the table dimension index that is related with the outputstatevariableindex (set -1 if there is not relation) 
   		 * 
   		 * It calculates the table dimension index that is related with the outputstatevariableindex (set -1 if there is not relation) 
   		 */
		void CalculateOutputTableDimensionIndex();

   		/*!
  		 * \brief It gets the maximum value inside the table.
  		 * 
  		 * It gets the maximum value inside the table.
  		 * 
  		 * \return The maximum value inside the table.
  		 */
  		float GetMaxElementInTable();
	private:

   		/*!
  		 * \brief Recrusive function used by GetMaxElementInTable().
  		 * 
  		 * Recrusive function used by GetMaxElementInTable().
		 * 
  		 * \param cpointer pointer to a dimension inside the table
		 * \param idim dimension inside the table
		 * \param ndims total number of dimensions.
  		 * 
  		 * \return The maximum value inside the table.
  		 */
		float CalculateMaxElementInTableRecursively(void **cpointer, int idim, int ndims);

	
		/*!
		 * Function used in function arrays.
		 */
		typedef float (NeuronModelTable::*function) (int index, VectorNeuronState * statevars);
   		
		/*!
		 * Function arrays.
		 */
		function funcArr[7];

   		/*!
   		 * Elements of the table.
   		 */
   		void *elems;
   
   		/*!
   		 * Number of dimensions.
   		 */
   		unsigned long ndims;
   		
   		/*!
   		 * Number of elements.
   		 */
   		unsigned long nelems;
   		
   		/*!
   		 * Table dimensions.
   		 */
   		TableDimension * dims;
   		
   		/*!
   		 * Interpolation of some dimension.
   		 */
   		int interp;
   		
   		/*!
   		 * The first interpolated dimension.
   		 */
   		int firstintdim;

   		/*!
   		 * State variable index that store this table (time it is not a state variable).
   		 */
		int outputstatevariableindex;


		int outputtabledimensionindex;


		float maxtimecoordenate;
		float inv_maxtimecoordenate;


  		
   		/*!
   		 * \brief It generates the virtual coordinates of the table.
   		 * 
   		 * This funcion generates the virtual coordinates of the table. That is necessary
   		 * for correctly accessing to the interpolated dimensions.
   		 * 
   		 * \throw EDLUTException If something wrong happens.
   		 */
   		void GenerateVirtualCoordinates() throw (EDLUTException);
   		


   		/*!
   		 * \brief It gets a table value without interpolation.
   		 * 
   		 * It gets a table value without interpolation with concrete state variables.
   		 * 
   		 * \param statevars State variables of the neuron.
   		 * 
   		 * \return The table value with that state.
   		 */
   		float TableAccessDirectDesviation(int index, VectorNeuronState * statevars);
   		
   		/*!
   		 * \brief It gets a table value with bilinear interpolation.
   		 * 
   		 * It gets a table value with bilinear interpolation with concrete state variables.
   		 * 
   		 * \param statevars State variables of the neuron.
   		 * 
   		 * \return The table value with that state.
   		 */
   		float TableAccessInterpBi(int index, VectorNeuronState * statevars);
   		
   		/*!
   		 * \brief It gets a table value with linear interpolation.
   		 * 
   		 * It gets a table value with linear interpolation with concrete state variables.
   		 * 
   		 * \param statevars State variables of the neuron.
   		 * 
   		 * \return The table value with that state.
   		 */
   		float TableAccessInterpLi(int index, VectorNeuronState * statevars);
   		
   		/*!
   		 * \brief It gets a table value with linear-ex interpolation.
   		 * 
   		 * It gets a table value with linear-ex interpolation with concrete state variables.
   		 * 
   		 * \param statevars State variables of the neuron.
   		 * 
   		 * \return The table value with that state.
   		 */
   		float TableAccessInterpLiEx(int index, VectorNeuronState * statevars);
   		
   		/*!
   		 * \brief It gets a table value with linear interpolation from 2 different positions.
   		 * 
   		 * It gets a table value with linear interpolation from 2 different positions with concrete state variables.
   		 * 
   		 * \param statevars State variables of the neuron.
   		 * 
   		 * \return The table value with that state.
   		 */
   		float TableAccessInterp2Li(int index, VectorNeuronState * statevars);
   		
   		/*!
   		 * \brief It gets a table value with linear interpolation from n different positions.
   		 * 
   		 * It gets a table value with linear interpolation from n different positions with concrete state variables.
   		 * 
   		 * \param statevars State variables of the neuron.
   		 * 
   		 * \return The table value with that state.
   		 */
   		float TableAccessInterpNLi(int index, VectorNeuronState * statevars);

   	
};

#endif /*NEURONMODELTABLE_H_*/

