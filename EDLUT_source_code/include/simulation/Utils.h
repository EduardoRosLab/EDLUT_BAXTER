/***************************************************************************
 *                           Utils.h                                       *
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

#ifndef UTILS_H_
#define UTILS_H_

/*!
 * \file Utils.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares functions for read and processs files.
 */

#include <cstdio>

#define COMMENT_CHAR '/'

/*!
 * \brief It skips the white lines, spaces and tab from the Currentline.
 * 
 * It skips the white lines from the Currentline in advance.
 * 
 * \param fh The file handler.
 * \param Currentline The currentline where we are readding.
 * 
 * \return True if white lines have been correctly cleared. False in other case.
 */
int skip_spaces(FILE *fh, long & Currentline);

/*!
 * \brief It skips the comment lines from the Currentline.
 * 
 * It skips the comment lines from the Currentline in advance.
 * 
 * \param fh The file handler.
 * \param Currentline The currentline where we are readding.
 */  
void skip_comments(FILE *fh, long & Currentline);


bool is_end_line(FILE *fh, long & Currentline);
  
#endif /*UTILS_H_*/
