/***************************************************************************
 *                           EDLUTFileException.h                          *
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

#ifndef EDLUTFILEEXCEPTION_H_
#define EDLUTFILEEXCEPTION_H_

/*!
 * \file EDLUTFileException.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts an exception in reading/writting files in the EDLUT simulation process.
 */

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "./EDLUTException.h"

using namespace std;

/*!
 * \class EDLUTFileException
 *
 * \brief Exception reading/writting files.
 *
 * This class abstract the behaviour of an exception in the EDLUT simulation process. This exception
 * could happen by reading/writting files. When
 * an exception is thrown, the simulation is stopped without results.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class EDLUTFileException: public EDLUTException {
	
	private:
		
		/*!
		 * File line where the error happens.
		 */
		long Currentline;

		/*!
		* File name where the error happens.
		*/
		const char * FileName;

		/*!
		* Set if the line must be plot.
		*/
		bool ShowLine;
		
	public:
	
		/*!
		 * \brief Class constructor.
		 * 
		 * Class constructor with parameters.
		 * 
		 * \param task The task code
		 * \param error The error code
		 * \param repair The repair code
		 * \param line Line where the error happened.
		 * \param file File name where the error happened.
		 * \param show set if the line must be shown
		 */ 
		EDLUTFileException(TASK_CODE task, ERROR_CODE error, REPAIR_CODE repair, long line, const char * file, bool show);
		
		/*!
		 * \brief It gets the line where the error happened.
		 * 
		 * It gets the line where the error happened.
		 * 
		 * \return The error line.
		 */
		long GetErrorLine() const;

		/*!
		* \brief It gets the file name where the error happened.
		*
		* It gets the file name where the error happened.
		*
		* \return The file name.
		*/
		const char * GetFileName() const;

		/*!
		* \brief It gets the ShowLine value.
		*
		* It gets the ShowLine value.
		*
		* \return The ShowLine value.
		*/
		bool GetShowLine() const;

		/*!
		 * \brief It prints in the standard error the error message of this exception.
		 * 
		 * It prints in the standard error the error message of this exception.
		 */
		void display_error() const;
		
};

/*!
 * \brief It prints the error message of this exception.
 * 
 * It prints the error message of this exception.
 * 
 * \param out The output stream where the message is printed.
 * \param Exception The exception to be printed.
 * 
 * \return The output stream.
 */
ostream & operator<< (ostream & out, EDLUTFileException Exception);

#endif /*EDLUTFILEEXCEPTION_H_*/
