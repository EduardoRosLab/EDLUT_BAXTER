/***************************************************************************
 *                           ConnectionException.h						   *
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

#ifndef CONNECTIONEXCEPTION_H_
#define CONNECTIONEXCEPTION_H_

/*!
 * \file ConnectionException.h
 *
 * \author Jesus Garrido
 * \date September 2008
 *
 * This file declares a class which abstracts an exception in the EDLUT TCP Connection Process.
 */

#include <string>
#include <iostream>

using namespace std;

/*!
 * \class ConnectionException
 *
 * \brief Exception in the connection process.
 *
 * This class abstract the behaviour of an exception in the connection process. This exception
 * could happen by different reasons...).
 *
 * \author Jesus Garrido
 * \date September 2008
 */
class ConnectionException {
	
	private:
		/*!
		 * Target Address.
		 */
		string Address;
		
		/*!
		 * Target Port.
		 */
		unsigned short Port;
		
		/*!
		 * Error message.
		 */
		string Message;
		
	public:
	
		/*!
		 * \brief Class constructor.
		 * 
		 * Class constructor with parameters.
		 * 
		 * \param TargetAddress Target address of the connection.
		 * \param TargetPort Target port of the connection.
		 * \param ErrorMessage Message describing the error.
		 */ 
		ConnectionException(string TargetAddress, unsigned short TargetPort, string ErrorMessage);
		
		/*!
		 * \brief It gets the connection target address.
		 * 
		 * It gets the connection target address.
		 * 
		 * \return The connection target address.
		 */ 
		string GetAddress() const;
		
		/*!
		 * \brief It gets the connection target port.
		 * 
		 * It gets the connection target port.
		 * 
		 * \return The connection target port.
		 */ 
		unsigned short GetPort() const;
		
		/*!
		 * \brief It gets the error message.
		 * 
		 * It gets the error that threw the error.
		 * 
		 * \return The error message.
		 */
		string GetErrorMsg() const;
		
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
ostream & operator<< (ostream & out, ConnectionException Exception);


#endif /*CONNECTIONEXCEPTION_H_*/
