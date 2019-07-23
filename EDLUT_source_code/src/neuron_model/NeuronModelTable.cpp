/***************************************************************************
 *                           NeuronModelTable.cpp                          *
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

#include "../../include/neuron_model/NeuronModelTable.h"

#include "../../include/simulation/Utils.h"

#include "../../include/neuron_model/VectorNeuronState.h"

#include <cfloat>
#include <cstdlib>

NeuronModelTable::TableDimension::TableDimension(): size(0), coord(0), vindex(0), voffset(0), vindex_voffset(0), vscale(0), inv_vscale(1), vfirst(0), vlast(0), statevar(0), interp(0), nextintdim(0) {
	
}
   				
NeuronModelTable::TableDimension::~TableDimension(){
	if (coord!=0) {
		delete [] coord;
	}
   			
	if (vindex!=0) {
		delete [] vindex;
	}
    
	if (voffset!=0) {
		delete [] voffset;
	}

	if (vindex_voffset!=0) {
		delete [] vindex_voffset;
	}
}

NeuronModelTable::NeuronModelTable(): elems(0), ndims(0), nelems(0), dims(0), interp(0), firstintdim(0), outputstatevariableindex(-1),outputtabledimensionindex(-1),maxtimecoordenate(-1),inv_maxtimecoordenate(-1){
	//Array of functions that include all the posible table access techniques.
	funcArr[0]=&NeuronModelTable::TableAccessDirect;
	funcArr[1]=&NeuronModelTable::TableAccessDirectDesviation;
	funcArr[2]=&NeuronModelTable::TableAccessInterpBi;
    funcArr[3]=&NeuronModelTable::TableAccessInterpLi;
	funcArr[4]=&NeuronModelTable::TableAccessInterpLiEx;
	funcArr[5]=&NeuronModelTable::TableAccessInterp2Li;
    funcArr[6]=&NeuronModelTable::TableAccessInterpNLi;

}
 
NeuronModelTable::~NeuronModelTable(){
	if (elems!=0) {
		free(elems);
	}

	if (dims!=0) {
		delete [] dims;
	}
}

int NeuronModelTable::TableDimension::table_indcomp(float coo){
	int index=(coo - this->vfirst) * this->inv_vscale + 0.49;
	int result=0;
	if(coo>this->vlast){
		result=this->size-1;
	}else if(coo>this->vfirst){
		result=*(this->vindex+index);
	}
	return result;			
}

int NeuronModelTable::TableDimension::table_indcomp2(float coo){
	int result=0;
	if(coo>this->vlast){
		result=this->size-1;
	}else if(coo>this->vfirst){
		float index=(coo - this->vfirst) * this->inv_vscale + 0.5;
		//result=*(this->vindex+(int)(index + *(this->voffset+(int)index)));
		result=*(this->vindex_voffset+2*(int)(index + *(this->vindex_voffset+1+2*(int)index)));
	}
	return result;			
}


int NeuronModelTable::TableDimension::table_ind_int(float coo){
	float index=(coo - this->vfirst) * this->inv_vscale;
	//result=*(this->vindex+(int)(index + *(this->voffset+(int)index)));
	return *(this->vindex_voffset+2*(int)(index + *(this->vindex_voffset+1+2*(int)index)));
}

float NeuronModelTable::TableDimension::check_range(float value){
	float new_value=value;
	if(this->coord[0]>value){
		new_value=coord[0];
	}else if(this->coord[this->size-1]<value){
		new_value=this->coord[this->size-1];
	}

	return new_value;		
}


void NeuronModelTable::SetOutputStateVariableIndex(int newoutputstatevariableindex){
	outputstatevariableindex=newoutputstatevariableindex;
}

int NeuronModelTable::GetOutputStateVariableIndex(){
	return outputstatevariableindex;
}

void NeuronModelTable::CalculateOutputTableDimensionIndex(){
	outputtabledimensionindex=-1;
	for(unsigned int idim=0;idim<this->ndims;idim++){
		if(this->outputstatevariableindex+1==(this->dims+idim)->statevar){///////////////////////
			outputtabledimensionindex=idim;
		}
	}
}



void NeuronModelTable::GenerateVirtualCoordinates() throw (EDLUTException){
	unsigned long idim,icoord;
	float minsca,sca,first,last,inv_minsca;
	unsigned int size;
	for(idim=0;idim<this->ndims;idim++){ // for each dimension of the table
		minsca=FLT_MAX;  // search for the minimum coordinate increment
		for(icoord=1;icoord < this->dims[idim].size;icoord++){
			sca=(this->dims[idim].coord[icoord] - this->dims[idim].coord[icoord-1]);
			if(sca < minsca && sca != 0.0){
				minsca=sca;
			}
		}

		inv_minsca=1.0f/minsca;
		
		first=this->dims[idim].coord[0];
		last=this->dims[idim].coord[this->dims[idim].size-1];
		this->dims[idim].vfirst=first;
		this->dims[idim].vlast=last;
		this->dims[idim].vscale=minsca;
		this->dims[idim].inv_vscale=inv_minsca;
		size=unsigned((last-first)/minsca+2);
		this->dims[idim].vindex=(int *) new int [size];
		this->dims[idim].voffset=(float *) new float [size];
		this->dims[idim].vindex_voffset=(float *) new float [2*size];
		if(this->dims[idim].vindex && this->dims[idim].voffset){  // create virtual coordinates
			unsigned int ivind,ipos;
			float coffset;
			ipos=0;          // dimension coordinates must grow monotonously
			for(ivind=0;ivind<size;ivind++){
				if(this->dims[idim].interp>1){  // interpolated dimension
					if(ipos+2 < this->dims[idim].size && this->dims[idim].coord[ipos+1] < ivind*minsca+first){
						ipos++;
					}
					
					if(ipos+1 <  this->dims[idim].size){
						coffset=1.0-((double) this->dims[idim].coord[ipos+1] - ((double)ivind*minsca+first))*inv_minsca;
						
						if(coffset < 0.0){
							coffset=0.0;
						}
					}else{
						coffset=0.0;
					}
				}else{  // non interpolated dimension
					if(ipos+1 < this->dims[idim].size && (this->dims[idim].coord[ipos]+this->dims[idim].coord[ipos+1])*0.5 < ivind*minsca+first){
						ipos++;
					}
					
					coffset=0.0;
					
					if(ipos+1 < this->dims[idim].size){
						coffset=((double)ivind*minsca+first)*inv_minsca + 0.5 - ((double)this->dims[idim].coord[ipos]+this->dims[idim].coord[ipos+1])*0.5*inv_minsca;
						if(coffset < 0.0){
							coffset=0.0;
						}
					}
	
					if(ipos > 0 && coffset == 0.0){
						coffset=((double)ivind*minsca+first)*inv_minsca - 0.5 - ((double)this->dims[idim].coord[ipos]+this->dims[idim].coord[ipos-1])*0.5*inv_minsca;
						
						if(coffset > 0.0){
							coffset=0.0;
						}
					}
				}
				this->dims[idim].voffset[ivind]=coffset;
				this->dims[idim].vindex[ivind]=ipos;

				this->dims[idim].vindex_voffset[2*ivind]=ipos;
				this->dims[idim].vindex_voffset[2*ivind+1]=coffset;
			}
		}else{
			delete [] this->dims[idim].vindex;
			this->dims[idim].vindex=0;
			delete [] this->dims[idim].voffset;
			this->dims[idim].voffset=0;
			break;
		}
	}
	
	if(idim != this->ndims){
		throw EDLUTException(TASK_NEURON_MODEL_TABLE, ERROR_NEURON_MODEL_TABLE_ALLOCATE, REPAIR_NEURON_MODEL_TABLE_ALLOCATE);
	}
	
}

void NeuronModelTable::TableInfo()
{
   unsigned long idim;
   printf("Number of elements: %lu\tNumber of dimensions: %lu\n",this->nelems,this->ndims);
   for(idim=0;idim < this->ndims;idim++){
      printf("Dimension %lu: size %lu vsize %i vscale %g coords: ",idim,this->dims[idim].size,(int)((this->dims[idim].coord[this->dims[idim].size-1]-this->dims[idim].coord[0])*this->dims[idim].inv_vscale+2),this->dims[idim].vscale);
      if(this->dims[idim].size>0){
         printf("[%g, %g]",this->dims[idim].coord[0],this->dims[idim].coord[this->dims[idim].size-1]);
      }
      printf("\n");
   }
}


void NeuronModelTable::LoadTable(FILE *fd) throw (EDLUTException){
	void **elems;
	unsigned long idim,totsize,vecsize;
	uint64_t nelems;
	 
	if(fread(&(nelems),sizeof(uint64_t),1,fd)==1){
		this->nelems = nelems;
		if(this->nelems>0 && (uint64_t)(this->nelems)==nelems){
			uint64_t ndims;
			if(fread(&ndims,sizeof(uint64_t),1,fd)==1){
            	this->ndims=ndims;

				vecsize=1L;
				totsize=0L;
				for(idim=0;idim < this->ndims;idim++){
					uint64_t dsize;
               		if(fread(&dsize,sizeof(uint64_t),1,fd)==1){
                  		this->dims[idim].size=dsize;
						vecsize*=this->dims[idim].size;
						totsize+=vecsize;
						this->dims[idim].coord=(float *) new float [this->dims[idim].size];
						if(this->dims[idim].coord){
							if(fread(this->dims[idim].coord,sizeof(float),this->dims[idim].size,fd)!=this->dims[idim].size){
								throw EDLUTException(TASK_NEURON_MODEL_TABLE_LOAD, ERROR_NEURON_MODEL_TABLE_NOT_ENOUGH_DATA, REPAIR_NEURON_MODEL_TABLE_NOT_ENOUGH_DATA);
								break;
							}
							//Calulate the maximum value of time that evaluate this table
							if(this->dims[idim].statevar==0){
								maxtimecoordenate=this->dims[idim].coord[this->dims[idim].size-1];
								inv_maxtimecoordenate=1.0f/maxtimecoordenate;
							}
						}else{
							throw EDLUTException(TASK_NEURON_MODEL_TABLE_LOAD, ERROR_NEURON_MODEL_TABLE_ALLOCATE, REPAIR_NEURON_MODEL_TABLE_ALLOCATE);
							break;
						}
					}else{
						throw EDLUTException(TASK_NEURON_MODEL_TABLE_LOAD, ERROR_NEURON_MODEL_TABLE_NOT_ENOUGH_DATA, REPAIR_NEURON_MODEL_TABLE_NOT_ENOUGH_DATA);
						break;
					}
				}
				
				if(idim==this->ndims){
					if(this->nelems != vecsize){
						char msgbuf[80];
						sprintf(msgbuf,"Inconsisten table (%lu elements expected)",vecsize);
						//show_info(msgbuf);
					}
					
					elems=(void **) malloc((totsize-this->nelems)*sizeof(void *)+this->nelems*sizeof(float));
					if(elems){
						unsigned long i;
						if(fread(elems+totsize-this->nelems,sizeof(float),this->nelems,fd) == this->nelems){
							vecsize=1L;
							totsize=0L;
							for(idim=0;idim < this->ndims-1;idim++){
								long relpos;
                        		void *basepos;
                        		vecsize*=this->dims[idim].size;
								for(i=0;i < vecsize;i++){
									relpos=i*this->dims[idim+1].size;
                           			basepos=elems+totsize+vecsize;
                           			*(elems+totsize+i)=(idim+1 < this->ndims-1)?(void **)basepos+relpos:(void **)((float *)basepos+relpos);
								}
								totsize+=vecsize;
							}
							
							this->elems=elems;
							GenerateVirtualCoordinates();
						}else{
							throw EDLUTException(TASK_NEURON_MODEL_TABLE_LOAD, ERROR_NEURON_MODEL_TABLE_NOT_ENOUGH_DATA, REPAIR_NEURON_MODEL_TABLE_NOT_ENOUGH_DATA);
						}
					}else{
						throw EDLUTException(TASK_NEURON_MODEL_TABLE_LOAD, ERROR_NEURON_MODEL_TABLE_ALLOCATE, REPAIR_NEURON_MODEL_TABLE_ALLOCATE);
					}
				}
			}else{
				throw EDLUTException(TASK_NEURON_MODEL_TABLE_LOAD, ERROR_NEURON_MODEL_TABLE_NOT_ENOUGH_DATA, REPAIR_NEURON_MODEL_TABLE_NOT_ENOUGH_DATA);
			}
		}else{
			(nelems>0) ? throw EDLUTException(TASK_NEURON_MODEL_TABLE_LOAD, ERROR_NEURON_MODEL_TABLE_TOO_BIG, REPAIR_NEURON_MODEL_TABLE_NOT_ENOUGH_DATA) : throw EDLUTException(TASK_NEURON_MODEL_TABLE_LOAD, ERROR_NEURON_MODEL_TABLE_EMPTY, REPAIR_NEURON_MODEL_TABLE_NOT_ENOUGH_DATA);
		}
	}else{
		throw EDLUTException(TASK_NEURON_MODEL_TABLE_LOAD, ERROR_NEURON_MODEL_TABLE_TABLE_NUMBER, REPAIR_NEURON_MODEL_TABLE_NOT_ENOUGH_DATA);
	}
}

void NeuronModelTable::LoadTableDescription(string ConfigFile, FILE *fh, long & Currentline) throw (EDLUTFileException){
	int previntdim;
	int nv;
	                          				
    this->elems=0;
	this->interp=0;
	skip_comments(fh,Currentline);
	
	if(fscanf(fh,"%li",&this->ndims)==1){
		this->dims = (TableDimension *) new TableDimension [this->ndims];

		previntdim=-1;
		this->firstintdim=-1;
		for(nv=this->ndims-1;nv>=0;nv--){
			this->dims[nv].coord=0;
			this->dims[nv].vindex=0;
			this->dims[nv].voffset=0;
			if(fscanf(fh,"%i",&this->dims[nv].statevar)!=1 || fscanf(fh,"%i",&this->dims[nv].interp)!=1){
				throw EDLUTFileException(TASK_NEURON_MODEL_TABLE_TABLES_STRUCTURE, ERROR_NEURON_MODEL_TABLE_VARIABLE_INDEX, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
			}else{
				if(this->dims[nv].interp>1){
					if(this->firstintdim==-1){
						this->firstintdim=nv;
					}else{
						this->dims[previntdim].nextintdim=previntdim-nv;
					}
					previntdim=nv;
					this->interp=this->dims[nv].interp;
   				}else{
					if(this->dims[nv].interp==1){
						this->interp=this->dims[nv].interp;
					}
    				this->dims[nv].nextintdim=1; // useless value
				}
			}
		}
			
		if(previntdim != -1){
			this->dims[previntdim].nextintdim=previntdim-nv;
		}
	}else{
		throw EDLUTFileException(TASK_NEURON_MODEL_TABLE_TABLES_STRUCTURE, ERROR_NEURON_MODEL_TABLE_DIMENSION_NUMBER, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
	}
}

float NeuronModelTable::GetMaxElementInTable(){
	unsigned int idim,tind;
	float elem;
	void **cpointer;
	NeuronModelTable::TableDimension *dim;
	cpointer=(void **)this->elems;
	return CalculateMaxElementInTableRecursively(cpointer, 0, this->ndims);
}

float NeuronModelTable::CalculateMaxElementInTableRecursively(void **cpointer, int idim, int ndims){
	NeuronModelTable::TableDimension *recrusivedim;
	void **recursivecpointer;
	recrusivedim = (this->dims + idim);
	float result=0.0;

	for(int tind=0; tind<recrusivedim->size; tind++){
		if(idim<ndims-1){
			recursivecpointer=(void **)*(cpointer+tind);
			float value=CalculateMaxElementInTableRecursively(recursivecpointer,idim+1,ndims);
			if(value>result){
				result=value;
			}
		}else{
			float value=*(((float *)cpointer)+tind);
			if(value>result){
				result=value;
			}
		}
	}
	return result;
}


const NeuronModelTable::TableDimension * NeuronModelTable::GetDimensionAt(int index) const{
	return this->dims+index;	
}
  		
void NeuronModelTable::SetDimensionAt(int index, NeuronModelTable::TableDimension Dimension){
	this->dims[index]=Dimension;	
}
  		
unsigned long NeuronModelTable::GetDimensionNumber() const{
	return this->ndims;
}
  		
float NeuronModelTable::GetElementAt(int index) const{
	return ((float *) this->elems)[index];
}
  		
void NeuronModelTable::SetElementAt(int index, float Element){
	((float *) this->elems)[index] = Element;
}  		

unsigned long NeuronModelTable::GetElementsNumber() const{
	return this->nelems;
}

int NeuronModelTable::GetInterpolation() const{
	return this->interp;
}
  		
int NeuronModelTable::GetFirstInterpolation() const{
	return this->firstintdim;
}

float NeuronModelTable::TableAccessDirect(int index, VectorNeuronState * statevars){
	unsigned int idim,tind;
	float elem;
	void **cpointer;
	NeuronModelTable::TableDimension *dim;
	cpointer=(void **)this->elems;
	float * VarValues=statevars->GetStateVariableAt(index);
	for(idim=0;idim<this->ndims-1;idim++){
		dim=this->dims+idim;
		tind=dim->table_indcomp2(VarValues[dim->statevar]);
		cpointer=(void **)*(cpointer+tind);
	}
	dim=this->dims+idim;
	tind=dim->table_indcomp2(VarValues[dim->statevar]);
	elem=*(((float *)cpointer)+tind);
	return(elem);
}

float NeuronModelTable::TableAccessDirectDesviation(int index, VectorNeuronState * statevars){

	unsigned int idim,tind;
	float elem;
	void **cpointer;
	NeuronModelTable::TableDimension *dim;
	cpointer=(void **)this->elems;
	float * VarValues=statevars->GetStateVariableAt(index);

	if(VarValues[0]>maxtimecoordenate || outputtabledimensionindex==-1){
		for(idim=0;idim<this->ndims-1;idim++){
			dim=this->dims+idim;
			tind=dim->table_indcomp2(VarValues[dim->statevar]);
			cpointer=(void **)*(cpointer+tind);
		}
		dim=this->dims+idim;
		tind=dim->table_indcomp2(VarValues[dim->statevar]);
		elem=*(((float *)cpointer)+tind);
	}else{
		float desviation=0;
		for(idim=0;idim<this->ndims-1;idim++){
			dim=this->dims+idim;
			tind=dim->table_indcomp2(VarValues[dim->statevar]);
			if(outputtabledimensionindex==idim){
				desviation=VarValues[dim->statevar]-dim->coord[tind];
			}
			cpointer=(void **)*(cpointer+tind);
		}
		dim=this->dims+idim;
		tind=dim->table_indcomp2(VarValues[dim->statevar]);
		if(outputtabledimensionindex==idim){
			desviation=VarValues[dim->statevar]-dim->coord[tind];
		}
		elem=*(((float *)cpointer)+tind);
	

		desviation*=(maxtimecoordenate-VarValues[0])*inv_maxtimecoordenate;
		//incremente output value in interpolated desviation
		elem+=desviation;

		//check range
		elem=(this->dims+outputtabledimensionindex)->check_range(elem);

	}
	return(elem);
}


// Bilineal interpolation
float NeuronModelTable::TableAccessInterpBi(int index, VectorNeuronState * statevars){
	int idim;
	float elem,*coord,*coords;
	NeuronModelTable *tab;
	NeuronModelTable::TableDimension *dim;
	int intstate[MAXSTATEVARS]={0};
	float subints[MAXSTATEVARS];
	float coeints[MAXSTATEVARS];
	void  **dpointers[MAXSTATEVARS];
	int tableinds[MAXSTATEVARS];

	tab=this;
	dpointers[0]=(void **)tab->elems;
	coord=statevars->GetStateVariableAt(index);

	for(idim=0;idim<tab->ndims;idim++){
		dim=&tab->dims[idim];
		if(dim->interp>1){
			coords=dim->coord;
			if(coord[dim->statevar]>dim->vlast){
				tableinds[idim]=dim->size-2;
				coeints[idim]=1;
			}else{
				if(coord[dim->statevar]<dim->vfirst){
					tableinds[idim]=0;
					coeints[idim]=0;
				}else{
					tableinds[idim]=dim->table_ind_int(coord[dim->statevar]);
					coeints[idim]=((coord[dim->statevar]-coords[tableinds[idim]])/(coords[tableinds[idim]+1]-coords[tableinds[idim]]));
				}
			}
		} else {
			tableinds[idim]=dim->table_indcomp2(coord[dim->statevar]);
		}
	}
	
	idim=0;
	do{
		for(;idim<tab->ndims-1;idim++){
			dpointers[idim+1]=(void **)dpointers[idim][tableinds[idim]+intstate[idim]];
		}
		
		elem=((float *)dpointers[idim])[tableinds[idim]+intstate[idim]];

		for(idim=tab->firstintdim;idim>=0;idim-=tab->dims[idim].nextintdim){


			intstate[idim]=!intstate[idim];
			if(intstate[idim]){
				subints[idim]=elem;
				break;
			}else{
				elem=subints[idim]=subints[idim]+(elem-subints[idim])*coeints[idim];
			}
		}
	} while(idim>=0);
   
	return(elem);
}


// Lineal interpolation
float NeuronModelTable::TableAccessInterpLi(int index, VectorNeuronState * statevars){
	int idim,iidim;
	float elem,elemi,elem0,*coord,*coords;
	NeuronModelTable *tab;
	NeuronModelTable::TableDimension *dim;
	float coeints[MAXSTATEVARS];
	void **dpointers[MAXSTATEVARS],**dpointer;
	int tableinds[MAXSTATEVARS];

	tab=this;
	dpointers[0]=(void **)tab->elems;
	coord=statevars->GetStateVariableAt(index);

	for(idim=0;idim<tab->ndims;idim++){
		dim=&tab->dims[idim];
		if(dim->interp>1){
			coords=dim->coord;
			if(coord[dim->statevar]>dim->vlast){
				tableinds[idim]=dim->size-2;
				coeints[idim]=1;
			}else{
				if(coord[dim->statevar]<dim->vfirst){
					tableinds[idim]=0;
					coeints[idim]=0;
				}else{
					tableinds[idim]=dim->table_ind_int(coord[dim->statevar]);
					coeints[idim]=((coord[dim->statevar]-coords[tableinds[idim]])/(coords[tableinds[idim]+1]-coords[tableinds[idim]]));
				}
			}
		}else{
			tableinds[idim]=dim->table_indcomp2(coord[dim->statevar]);
		}
	}
	
	for(idim=0;idim<tab->ndims-1;idim++){
		dpointers[idim+1]=(void **)dpointers[idim][tableinds[idim]];
	}
	
	elemi=elem0=((float *)dpointers[idim])[tableinds[idim]];

	for(iidim=tab->firstintdim;iidim>=0;iidim-=tab->dims[iidim].nextintdim){
		dpointer=dpointers[iidim];
		for(idim=iidim;idim<tab->ndims-1;idim++){
			dpointer=(void **)dpointer[tableinds[idim]+(idim==iidim)];
		}
		elem=((float *)dpointer)[tableinds[idim]+(idim==iidim)];
		elemi+=(elem-elem0)*coeints[iidim];
	}
	
	return(elemi);
}

// Lineal interpolation-extrapolation
float NeuronModelTable::TableAccessInterpLiEx(int index, VectorNeuronState * statevars){
	int idim,iidim;
	float elem,elemi,elem0,*coord,*coords;
	NeuronModelTable *tab;
	NeuronModelTable::TableDimension *dim;
	float coeints[MAXSTATEVARS];
	void **dpointers[MAXSTATEVARS],**dpointer;
	int tableinds[MAXSTATEVARS];

	tab=this;
	dpointers[0]=(void **)tab->elems;
	coord=statevars->GetStateVariableAt(index);

	for(idim=0;idim<tab->ndims;idim++){
		dim=&tab->dims[idim];
		if(dim->interp>1){
			coords=dim->coord;
			if(coord[dim->statevar]>dim->vlast){
				tableinds[idim]=dim->size-2;
			}else{
				if(coord[dim->statevar]<dim->vfirst){
					tableinds[idim]=0;
				}else{
					tableinds[idim]=dim->table_ind_int(coord[dim->statevar]);
				}
			}
			
			coeints[idim]=((coord[dim->statevar]-coords[tableinds[idim]])/(coords[tableinds[idim]+1]-coords[tableinds[idim]]));
		}else{
			tableinds[idim]=dim->table_indcomp2(coord[dim->statevar]);
		}
	}
	
	for(idim=0;idim<tab->ndims-1;idim++){
		dpointers[idim+1]=(void **)dpointers[idim][tableinds[idim]];
	}
   
	elemi=elem0=((float *)dpointers[idim])[tableinds[idim]];

	for(iidim=tab->firstintdim;iidim>=0;iidim-=tab->dims[iidim].nextintdim){
		dpointer=dpointers[iidim];
		for(idim=iidim;idim<tab->ndims-1;idim++){
			dpointer=(void **)dpointer[tableinds[idim]+(idim==iidim)];
		}
		elem=((float *)dpointer)[tableinds[idim]+(idim==iidim)];
		elemi+=(elem-elem0)*coeints[iidim];
	}
	
	return(elemi);
}

// 2-position lineal interpolation
float NeuronModelTable::TableAccessInterp2Li(int index, VectorNeuronState * statevars){
	int idim,iidim,nintdims,zpos;
	float elem,elemi,elem0,avepos,*coord,*coords;
	NeuronModelTable *tab;
	NeuronModelTable::TableDimension *dim;
	float coeints[MAXSTATEVARS];
	void **dpointers[MAXSTATEVARS],**dpointer;
	int tableinds[MAXSTATEVARS];

	tab=this;
	dpointers[0]=(void **)tab->elems;
	coord=statevars->GetStateVariableAt(index);

	avepos=0;
	nintdims=0;
	for(idim=0;idim<tab->ndims;idim++){
		dim=&tab->dims[idim];
		if(dim->interp>1){
			coords=dim->coord;
			if(coord[dim->statevar]>dim->vlast){
				tableinds[idim]=dim->size-2;
				coeints[idim]=1;
			}else{
				if(coord[dim->statevar]<dim->vfirst){
					tableinds[idim]=0;
					coeints[idim]=0;
				}else{
					tableinds[idim]=dim->table_ind_int(coord[dim->statevar]);
					coeints[idim]=((coord[dim->statevar]-coords[tableinds[idim]])/(coords[tableinds[idim]+1]-coords[tableinds[idim]]));
				}
			}
			
			avepos+=coeints[idim];
			nintdims++;
		}else{
			tableinds[idim]=dim->table_indcomp2(coord[dim->statevar]);
		}
	}
    
    zpos=(avepos/nintdims)>0.5;
    if(zpos){
    	for(iidim=tab->firstintdim;iidim>=0;iidim-=tab->dims[iidim].nextintdim){
    		tableinds[iidim]++;
    		coeints[iidim]=coeints[iidim]-1;
    	}
    }
    
    zpos=zpos*-2+1; // pos=1 or -1
    for(idim=0;idim<tab->ndims-1;idim++){
    	dpointers[idim+1]=(void **)dpointers[idim][tableinds[idim]];
    }
    
    elemi=elem0=((float *)dpointers[idim])[tableinds[idim]];

	for(iidim=tab->firstintdim;iidim>=0;iidim-=tab->dims[iidim].nextintdim){
		dpointer=dpointers[iidim];
		for(idim=iidim;idim<tab->ndims-1;idim++){
			dpointer=(void **)dpointer[tableinds[idim]+zpos*(idim==iidim)];
		}
		
		elem=((float *)dpointer)[tableinds[idim]+zpos*(idim==iidim)];
		elemi+=(elem-elem0)*coeints[iidim];
	}
   
	return(elemi);
}

// n-position lineal interpolation
float NeuronModelTable::TableAccessInterpNLi(int index, VectorNeuronState * statevars){
	int idim;
	int iidim;
	float elem,elemi,elem0,*coord,*coords;
	NeuronModelTable *tab;
	NeuronModelTable::TableDimension *dim;
	int intstate[MAXSTATEVARS];
	float coeints[MAXSTATEVARS];
	void **dpointers[MAXSTATEVARS],**dpointer;
	int tableinds[MAXSTATEVARS];

	tab=this;
	dpointers[0]=(void **)tab->elems;
	coord=statevars->GetStateVariableAt(index);


	for(idim=0;idim<tab->ndims;idim++){
		dim=&tab->dims[idim];
		if(dim->interp>1){
			coords=dim->coord;
			if(coord[dim->statevar]>dim->vlast){
				tableinds[idim]=dim->size-2;
				coeints[idim]=1;
			}else{
				if(coord[dim->statevar]<dim->vfirst){
					tableinds[idim]=0;
					coeints[idim]=0;
				}else{
					tableinds[idim]=dim->table_ind_int(coord[dim->statevar]);
					coeints[idim]=((coord[dim->statevar]-coords[tableinds[idim]])/(coords[tableinds[idim]+1]-coords[tableinds[idim]]));
				}
			}
			
			if(coeints[idim]>0.5){
				coeints[idim]=1-coeints[idim];
				intstate[idim]=1;
			}else{
				intstate[idim]=0;
			}
		}else{
			tableinds[idim]=dim->table_indcomp2(coord[dim->statevar]);
			intstate[idim]=0;
        }
	}
   
	for(idim=0;idim<tab->ndims-1;idim++){
		dpointers[idim+1]=(void **)dpointers[idim][tableinds[idim]+intstate[idim]];
	}
	
	elemi=elem0=((float *)dpointers[idim])[tableinds[idim]+intstate[idim]];

	for(iidim=tab->firstintdim;iidim>=0;iidim-=tab->dims[iidim].nextintdim){
		dpointer=dpointers[iidim];
		for(idim=iidim;idim<tab->ndims-1;idim++){
			dpointer=(void **)dpointer[tableinds[idim]+(intstate[idim]^(idim==(unsigned int)iidim))];
		}
		
		elem=((float *)dpointer)[tableinds[idim]+(intstate[idim]^(idim==(unsigned int)iidim))];
		elemi+=(elem-elem0)*coeints[iidim];
	}
	
	return(elemi);
}

float NeuronModelTable::TableAccess(int index, VectorNeuronState * statevars){
	return(this->*funcArr[this->interp])(index,statevars);
}
