/***************************************************************************
 *                           CDSocket.h                                    *
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

#ifndef CD_SOCKET_H
#define CD_SOCKET_H

#include <iostream>

#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>

#ifdef _WIN32
	#include <winsock2.h>
#endif

#include "./CommunicationDevice.h"

using namespace std;

#define CLIENT 0
#define SERVER 1

//( Cd_Socket

/*!
 * \brief A socket used to communicate with another program
 *
 * \author Christian Boucheny
 * \author Jesús Garrido
 *
 * \note Modified on November 2010 to add windows socket compatibility.
 **/
class CdSocket : public CommunicationDevice {
public:
  /*!
   *
   * \brief Constructor for which parameters of the tcp connection are stored in an external file 
   * 
   * \param status CLIENT or SERVER 
   * \param server_address address of the server host
   * \param tcp_port tcp_port to connect
   *
   **/
  CdSocket(unsigned short status, string server_address,unsigned short tcp_port);

  /*!
   * \brief Default destructor
   *
   **/
  ~CdSocket();

  /*!
   *
   * \brief Send a block of data
   * 
   * \param   buffer       data to send. Need a cast to char*
   * \param   buffer_size  size of the packet
   *
   * \return  the error signal 
   *
   **/
  int sendBuffer(void* buffer,int buffer_size);

  /*!
   *
   * \brief Receive a block of data
   * 
   * \param   buffer       data to send
   * \param   buffer_size  size of the packet
   *
   * \return  the error signal 
   *
   **/
  int receiveBuffer(void* buffer,int buffer_size);


protected:
  /*!
   *
   * initialize the socket connections with parameters defined in constructor 
   * 
   *
   **/
  void initializeSocket();

#ifdef _WIN32
  // Windows Variables
  /*!
   *
   * Number of socket instances in order to load or unload the windows socket library.
   *
   */
  static unsigned int SocketInstances;

  /*!
   *
   * Socket identification.
   *
   */
  SOCKET socket_fd;

#else

  /*!
   *
   * The socket itself
   * 
   **/
  int socket_fd;

#endif

  /*!
   *
   * host address of the server
   * 
   **/
  string serv_host_addr;

  /*!
   *
   * tcp port of the server
   * 
   **/
  unsigned short serv_tcp_port;

  /*!
   *
   * is it a client or server ? 0 = CLIENT / 1 = SERVER
   * 
   **/
  unsigned short status;

  /*!
   *
   * return value of called I-O functions 
   * 
   **/
  int ret_status;
}
;

//) Cd_Socket


#endif

//) cd_socket.hpp
