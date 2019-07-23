%THIS FILE TRY TO COMPRESS THE DECLARATION OF THE SYNAPSES IN A NETWORK
%FILE. THE INPUT FILE MUST CONTAIN ONLY THE LIST OF SYNAPSES. ADDITIONALLY,
%THE USER MUST PRINT THE NUMBER OF SYNAPSES TO COMPUTE IN THE VARIABLE
%"N_connections"

%INSERT THE NAME OF THE INPUT AND OUTPUT FILE.
Name_input = 'network_I.net';
Name_output = 'network_Ob.net';

%iNSERTE THE NOMBER OF SYNAPSES TO BE COMPUTED
N_connections=8830003;

%%

tic

F_input=fopen(Name_input, 'r');
F_output=fopen(Name_output, 'w');


count_connection=0;
iteration=0;
type=-1;

while count_connection < N_connections
    index=fscanf(F_input,'%d', 5);
    propagation=fscanf(F_input,'%g', 2);
    connection_type=fscanf(F_input,'%d', 1);
    max_weight=fscanf(F_input,'%g', 1);
    learning_rule=fscanf(F_input,'%d', 1);
    
    %it is the first synapse of a groupe
    if (iteration==0)
        %It is the last synapse.
        if (count_connection+index(2)*index(4)*index(5))==N_connections
            fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', index(1),index(2),index(3),index(4),index(5),propagation(1), propagation(2), connection_type, max_weight, learning_rule);
            count_connection=count_connection+index(2)*index(4)*index(5);
        else
            %The reads line is only for a synapse.
            if (index(2)==1 && index(4)==1 && index(5)==1)
                initial_index=index;
                initial_propagation=propagation;
                initial_connection_type=connection_type;
                initial_max_weight=max_weight;
                initial_learning_rule=learning_rule;

                iteration=1;
            else
                fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', index(1),index(2),index(3),index(4),index(5),propagation(1), propagation(2), connection_type, max_weight, learning_rule);
                count_connection=count_connection+index(2)*index(4)*index(5);
                
                iteration=0;
            end
        end
        type=-1;
    else
        %the readed line is only for a synapse and has the same parameter
        %that the previous synapse.
        if(index(2)==1 && index(4)==1 && index(5)==1 && propagation(1)==initial_propagation(1) && connection_type==initial_connection_type && max_weight==initial_max_weight && learning_rule==initial_learning_rule)
            %The relation with the previous synapse was:
            
            %type=0 --> one source neuron conected to consecutive target
            %neurons
            if type==0
                if index(1)==initial_index(1) && index(3)==(initial_index(3)+initial_index(4))
                    initial_index(4)=initial_index(4)+1;
                    iteration=iteration+1;
                    if (count_connection+initial_index(2)*initial_index(4)*initial_index(5))==N_connections
                        fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', initial_index(1),initial_index(2),initial_index(3),initial_index(4),initial_index(5),initial_propagation(1), initial_propagation(2), initial_connection_type, initial_max_weight, initial_learning_rule);
                        count_connection=count_connection+initial_index(2)*initial_index(4)*initial_index(5);
                    end
                else
                    fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', initial_index(1),initial_index(2),initial_index(3),initial_index(4),initial_index(5),initial_propagation(1), initial_propagation(2), initial_connection_type, initial_max_weight, initial_learning_rule);
                    count_connection=count_connection+initial_index(2)*initial_index(4)*initial_index(5);
                    if index(2)==1 && index(4)==1 && index(5)==1
                        initial_index=index;
                        initial_propagation=propagation;
                        initial_connection_type=connection_type;
                        initial_max_weight=max_weight;
                        initial_learning_rule=learning_rule;

                        iteration=1;
                    else
                        fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', index(1),index(2),index(3),index(4),index(5),propagation(1), propagation(2), connection_type, max_weight, learning_rule);
                        count_connection=count_connection+index(2)*index(4)*index(5); 
                        iteration=0;
                    end
                    type=-2;
                end
            end
            
            %type=1 --> one target neuron conected to consecutive source
            %neurons
            if type==1
                if index(1)==(initial_index(1)+initial_index(2)) && index(3)==initial_index(3)
                    initial_index(2)=initial_index(2)+1;
                    iteration=iteration+1;
                    if (count_connection+initial_index(2)*initial_index(4)*initial_index(5))==N_connections
                        fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', initial_index(1),initial_index(2),initial_index(3),initial_index(4),initial_index(5),initial_propagation(1), initial_propagation(2), initial_connection_type, initial_max_weight, initial_learning_rule);
                        count_connection=count_connection+initial_index(2)*initial_index(4)*initial_index(5);
                    end
                else
                    fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', initial_index(1),initial_index(2),initial_index(3),initial_index(4),initial_index(5),initial_propagation(1), initial_propagation(2), initial_connection_type, initial_max_weight, initial_learning_rule);
                    count_connection=count_connection+initial_index(2)*initial_index(4)*initial_index(5);
                    if index(2)==1 && index(4)==1 && index(5)==1
                        initial_index=index;
                        initial_propagation=propagation;
                        initial_connection_type=connection_type;
                        initial_max_weight=max_weight;
                        initial_learning_rule=learning_rule;

                        iteration=1;
                     else
                        fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', index(1),index(2),index(3),index(4),index(5),propagation(1), propagation(2), connection_type, max_weight, learning_rule);
                        count_connection=count_connection+index(2)*index(4)*index(5); 
                        iteration=0;
                    end
                    type=-2;
                end
            end
            
            %type=2 --> consecutive source neurons conected to consecutive target
            %neurons
            if type==2
                if index(1)==(initial_index(1)+initial_index(5)) && index(3)==(initial_index(3)+initial_index(5))
                    initial_index(5)=initial_index(5)+1;
                    iteration=iteration+1;
                    if (count_connection+initial_index(2)*initial_index(4)*initial_index(5))==N_connections
                        fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', initial_index(1),initial_index(2),initial_index(3),initial_index(4),initial_index(5),initial_propagation(1), initial_propagation(2), initial_connection_type, initial_max_weight, initial_learning_rule);
                        count_connection=count_connection+initial_index(2)*initial_index(4)*initial_index(5);
                    end
                else
                    fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', initial_index(1),initial_index(2),initial_index(3),initial_index(4),initial_index(5),initial_propagation(1), initial_propagation(2), initial_connection_type, initial_max_weight, initial_learning_rule);
                    count_connection=count_connection+initial_index(2)*initial_index(4)*initial_index(5);
                    if index(2)==1 && index(4)==1 && index(5)==1
                        initial_index=index;
                        initial_propagation=propagation;
                        initial_connection_type=connection_type;
                        initial_max_weight=max_weight;
                        initial_learning_rule=learning_rule;

                        iteration=1;
                    else
                        fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', index(1),index(2),index(3),index(4),index(5),propagation(1), propagation(2), connection_type, max_weight, learning_rule);
                        count_connection=count_connection+index(2)*index(4)*index(5); 
                        iteration=0;
                    end
                    type=-2;
                end
            end
            
            %type=-1 --> the relation between synapses was unknown.
            if type==-1 
                %if it is of type 0.
                if index(1)==initial_index(1) && index(3)==(initial_index(3)+initial_index(4))
                    type=0;
                  
                    initial_index(4)=initial_index(4)+1;
                    iteration=iteration+1;
                    if (count_connection+initial_index(2)*initial_index(4)*initial_index(5))==N_connections
                        fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', initial_index(1),initial_index(2),initial_index(3),initial_index(4),initial_index(5),initial_propagation(1), initial_propagation(2), initial_connection_type, initial_max_weight, initial_learning_rule);
                        count_connection=count_connection+initial_index(2)*initial_index(4)*initial_index(5);
                    end
                
                %if it is of type 1.
                elseif index(1)==(initial_index(1)+initial_index(2)) && index(3)==initial_index(3)
                    type=1;
                  
                    initial_index(2)=initial_index(2)+1;
                    iteration=iteration+1;
                    if (count_connection+initial_index(2)*initial_index(4)*initial_index(5))==N_connections
                        fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', initial_index(1),initial_index(2),initial_index(3),initial_index(4),initial_index(5),initial_propagation(1), initial_propagation(2), initial_connection_type, initial_max_weight, initial_learning_rule);
                        count_connection=count_connection+initial_index(2)*initial_index(4)*initial_index(5);
                    end
                
                %if it is of type 2.
                elseif index(1)==(initial_index(1)+initial_index(5)) && index(3)==(initial_index(3)+initial_index(5))
                    type=2;
                  
                    initial_index(5)=initial_index(5)+1;
                    iteration=iteration+1;
                    if (count_connection+initial_index(2)*initial_index(4)*initial_index(5))==N_connections
                        fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', initial_index(1),initial_index(2),initial_index(3),initial_index(4),initial_index(5),initial_propagation(1), initial_propagation(2), initial_connection_type, initial_max_weight, initial_learning_rule);
                        count_connection=count_connection+initial_index(2)*initial_index(4)*initial_index(5);
                    end
                    
                %if it is of type unknow.
                else
                    fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', initial_index(1),initial_index(2),initial_index(3),initial_index(4),initial_index(5),initial_propagation(1), initial_propagation(2), initial_connection_type, initial_max_weight, initial_learning_rule);
                    count_connection=count_connection+initial_index(2)*initial_index(4)*initial_index(5);

                    if (count_connection+index(2)*index(4)*index(5))==N_connections
                        fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', index(1),index(2),index(3),index(4),index(5),propagation(1), propagation(2), connection_type, max_weight, learning_rule);
                        count_connection=count_connection+index(2)*index(4)*index(5);
                    else
                        initial_index=index;
                        initial_propagation=propagation;
                        initial_connection_type=connection_type;
                        initial_max_weight=max_weight;
                        initial_learning_rule=learning_rule;

                        iteration=1;
                        type=-1;
                    end
                end
            end
            
            %auxiliar state.
            if type ==-2
                type=-1;
            end
        else
            fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', initial_index(1),initial_index(2),initial_index(3),initial_index(4),initial_index(5),initial_propagation(1), initial_propagation(2), initial_connection_type, initial_max_weight, initial_learning_rule);
            count_connection=count_connection+initial_index(2)*initial_index(4)*initial_index(5);
            
            if (count_connection+index(2)*index(4)*index(5))==N_connections
                fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', index(1),index(2),index(3),index(4),index(5),propagation(1), propagation(2), connection_type, max_weight, learning_rule);
                count_connection=count_connection+index(2)*index(4)*index(5);
            else
                if index(2)==1 && index(4)==1 && index(5)==1
                    initial_index=index;
                    initial_propagation=propagation;
                    initial_connection_type=connection_type;
                    initial_max_weight=max_weight;
                    initial_learning_rule=learning_rule;

                    iteration=1;
                else
                    fprintf(F_output, '%d %d %d %d %d %g %g %d %g %d\n', index(1),index(2),index(3),index(4),index(5),propagation(1), propagation(2), connection_type, max_weight, learning_rule);
                    count_connection=count_connection+index(2)*index(4)*index(5); 
                    iteration=0;
                end
            end
            type=-1;
        end
    end   
end;

fclose(F_input);
fclose(F_output);

toc

