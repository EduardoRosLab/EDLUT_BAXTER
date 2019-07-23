/***************************************************************************
 *                           CdSocket.cpp                                  *
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

#include "../../include/communication/CdSocket.h"

#ifdef _WIN32
	#include <windows.h>
	#include <winsock2.h>

	#pragma comment(lib, "ws2_32.lib")

	unsigned int CdSocket::SocketInstances = 0;

#else
	// Linux socket includes
	#include <sys/types.h>
	#include <sys/socket.h>
	#include <sys/un.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
	#include <fcntl.h> // to set special server flags
	#include <unistd.h> // misc symbolic constants and types
	#include <netdb.h>
	#include <netinet/tcp.h>
#endif

#define SERVER_PATH "server"

CdSocket::CdSocket(unsigned short status, string server_address,unsigned short tcp_port){
	this->status = status;
	this->serv_host_addr = server_address;
	this->serv_tcp_port = tcp_port;

	initializeSocket();
}

CdSocket::~CdSocket(){
#ifdef _WIN32
	closesocket(socket_fd);
	SocketInstances--;
	if (SocketInstances==0){
		WSACleanup();
	}
#else
	close(socket_fd);
#endif
}

void CdSocket::initializeSocket()
{  
	cout << "Connecting to Server " << this->serv_host_addr << ":" << this->serv_tcp_port << endl;
#ifdef _WIN32
	if (SocketInstances==0){
		WSADATA wsaData;
		WORD version;

		int error;

		version = MAKEWORD(2,2);

		error = WSAStartup(version, &wsaData);

		/* check for error */
		if ( error != 0 )
		{
			cerr << "Problem when creating the windows socket v2.2" << endl;
			WSACleanup();
			exit(EXIT_FAILURE);
		}

		/* check for correct version */
		if ( LOBYTE( wsaData.wVersion ) != 2 || HIBYTE( wsaData.wVersion ) != 2 ){
		    /* incorrect WinSock version */
		    WSACleanup();
		    cerr << "Invalid version of windows socket - Not v2.2 available" << endl;
		    exit(EXIT_FAILURE);
		}

		/* WinSock has been initialized */

		SocketInstances ++;
	}

#endif

	/***************/
	/*** CLIENT  ***/
	/***************/
	if(this->status==CLIENT){
		socket_fd = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);

#ifdef _WIN32
		if(socket_fd == INVALID_SOCKET){
			cerr << "Problem when creating the windows socket: creation ... closing the simulation (CD_SOCKET)" << endl;
			WSACleanup();
			exit(EXIT_FAILURE);
		}
#else
		if(socket_fd == -1){
			cerr << "Problem when creating the socket: creation ... closing the simulation (CD_SOCKET)" << endl;
			exit(EXIT_FAILURE);
		}
#endif


		struct sockaddr_in sin;

		memset( &sin, 0, sizeof sin );
		struct sockaddr_in serv_addr;
		memset((char *)&serv_addr, 0, sizeof(serv_addr));
		serv_addr.sin_family = AF_INET;      
		serv_addr.sin_addr.s_addr = inet_addr(this->serv_host_addr.c_str());
		serv_addr.sin_port = htons(this->serv_tcp_port);
		int socket_connect = connect(socket_fd,(struct sockaddr *)&serv_addr, sizeof(serv_addr)); 

		if(socket_connect != 0){
			cerr << "Problem when connecting socket: connection ... closing the simulation ( (CD_SOCKET))" << endl;
			exit(EXIT_FAILURE);
		}

		struct protoent *p;
		int one=1;
		p = getprotobyname("tcp");
		setsockopt(socket_fd, p->p_proto, TCP_NODELAY, (const char*)&one, sizeof(one));
    }


	/***************/
	/*** SERVER  ***/
	/***************/
	if(this->status==SERVER){


		int clilen, ret;
		struct sockaddr_in cli_addr, serv_addr;
		memset((char *)&serv_addr, 0, sizeof(serv_addr));

#ifdef _WIN32
		SOCKET tmp_socket_fd = INVALID_SOCKET;
#else
		int tmp_socket_fd;
#endif

		tmp_socket_fd = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);

#ifdef _WIN32
		if(tmp_socket_fd == INVALID_SOCKET){
			cerr << "Problem when creating the windows socket: creation ... closing the simulation (CD_SOCKET)" << endl;
			WSACleanup();
			exit(EXIT_FAILURE);
		}
#else
		if(tmp_socket_fd == -1){
			cerr << "Problem when creating the socket: creation ... closing the simulation (CD_SOCKET)" << endl;
			exit(EXIT_FAILURE);
		}
#endif

		serv_addr.sin_family = AF_INET;
		serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
		serv_addr.sin_port = htons(this->serv_tcp_port);

		ret = bind(tmp_socket_fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
		if(ret!=0){
			cerr << "Problem when binding socket ... closing the simulation  (CD_SOCKET)" << endl;
			exit(EXIT_FAILURE);
		}
  
		ret = listen(tmp_socket_fd, 1);
		if(ret!=0){
			cerr << "Problem when listening socket ... closing the simulation (CD_SOCKET)" << endl;
			exit(EXIT_FAILURE);
		}


		clilen = sizeof(cli_addr);
		

#ifdef _WIN32
		socket_fd = accept(tmp_socket_fd, (struct sockaddr *) &cli_addr, &clilen);
		closesocket(tmp_socket_fd);		
#else
		socket_fd = accept(tmp_socket_fd, (struct sockaddr *) &cli_addr, (socklen_t *)&clilen);
		close(tmp_socket_fd);          // close original socket
#endif


#ifdef _WIN32
		if(socket_fd == INVALID_SOCKET){
			cerr << "Problem when creating the windows socket: Couldn't accept the external connection (CD_SOCKET)" << endl;
			WSACleanup();
			exit(EXIT_FAILURE);
		}
#else
		if(socket_fd == -1){
			cerr << "Problem when creating the socket: creation ... closing the simulation (CD_SOCKET)" << endl;
			exit(EXIT_FAILURE);
		}
#endif
		
		struct protoent *p;
		int one=1;
		p = getprotobyname("tcp");
		setsockopt(socket_fd, p->p_proto, TCP_NODELAY, (const char*)&one, sizeof(one));

		//linger ls = {1, 6000} ; 
		//setsockopt(socket_fd, p->p_proto, SO_LINGER, (const char *)&ls, sizeof(ls)) ;
	}

	cout << "Connection OK" << endl;
}



int CdSocket::receiveBuffer(void* buffer, int buffer_size){
	ret_status=0;
	for(int nbytes=0;nbytes<buffer_size;){
		/*** to be sure that everything is received ***/
		ret_status=recv(socket_fd,((char*)buffer)+nbytes,buffer_size-nbytes,0);
		if(ret_status > 0)
			nbytes+=ret_status;
		else
			break;
    }
	return(ret_status);
}

int CdSocket::sendBuffer(void* buffer, int buffer_size){
	return int(send(socket_fd,(char *)buffer,buffer_size,0));
}


