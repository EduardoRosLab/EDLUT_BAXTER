/***************************************************************************
 *                           PrintableObject.h                             *
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

#ifndef PRINTABLEOBJECT_H_
#define PRINTABLEOBJECT_H_

/*!
 * \file PrintableObject.h
 *
 * \author Jesus Garrido
 * \date February 2010
 *
 * This file declares a class declare the behaviour of an object which can be printed.
 */

#include <ostream>

using namespace std;

/*!
 * \class PrintableObject
 *
 * \brief Characteristics for printable object.
 *
 * Every object which can be printed must implement these functions.
 *
 * \author Jesus Garrido
 * \date February 2010
 */
class PrintableObject {
	public:

		/*!
		 * \brief Default destructor
		 *
		 * Virtual function. It has to be implemented in every inherited class.
		 */
		virtual ~PrintableObject() {};

		/*!
		 * \brief It prints the information of the object.
		 *
		 * It prints the information of the object.
		 *
		 * \param out The output stream where it prints the object to.
		 * \return The output stream.
		 */
		virtual ostream & PrintInfo(ostream & out) = 0;

};

#endif /* PRINTABLEOBJECT_H_ */
