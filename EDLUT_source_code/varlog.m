%VARLOG plot robot's the registered simulation state variables
%   VARLOG var_type reads the simulation state variables during the whole 
%   simulation from the simulation log file and plot them
%   VARLOG var_type t_ini and t_end only reads the registers from
%   t_ini to t_end
%   var_type is the variables (file columns) which must be plotted.
%   var_type must be: ti, in, st, to, tt, ou, er, le, ma
%    ti: Plot consumed computation-time information
%        (only one trajectory excution is plotted)
%    in: Plot the desired trajectory posistion, velocity and
%        acceleration (one execution)
%    st: Plot the desired trajectory versus the actual robot posistion,
%        velocity and acceleration trajectory
%    to: Plot the crude torque calculated by the inverse RNE algorith
%    tt: Plot the total applied torque (cerebellum corrective torque
%        plus crude inverse dynamics torque)
%    ou: Plot the the corrective cerebellar output torque
%    er: Plot the robot's performed error per joint
%    le: Plot the learning signal computed from the robot's error
%    ma: Plot the mean average error per trajectory execution
%   example to plot the robot state in the time interval 0 1: varlog st 0 1
function varlog(varargin)

LOG_FILE='vars.dat'; % log file name
single_exec_dur=1; % trajectoy duration in seconds
num_joints=3; % number of robot's joints
sim_slot_time=2e-3; % simulation slot length in seconds

nargs=nargin;
if nargin > 0
   v=varargin{1};
   if ~isequal(v,'ti') && ~isequal(v,'in') && ~isequal(v,'st') && ~isequal(v,'to') && ~isequal(v,'tt') && ~isequal(v,'ou') && ~isequal(v,'er') && ~isequal(v,'le') && ~isequal(v,'ma')
      display('Incorrect value of the first argument');
      nargs=0;
   end
end

switch(nargs)
   case 1
      disp(['loading log file: ' LOG_FILE ' completely']);
      disp(' 100%');
      vars=load(LOG_FILE);
   case 3
      start_time=str2num(varargin{2});
      end_time=str2num(varargin{3});
      disp(['Partially loading log file: ' LOG_FILE ' time from ' num2str(start_time) ' to ' num2str(end_time)]);
      fid = fopen(LOG_FILE,'rt');
      if fid ~= -1
         fseek(fid,-1,'eof');
         filelength=ftell(fid); % get log file size
         findline_backwards(fid); % set file reading pointer to the line start
         last_line_txt=fgetl(fid); % load last register
         last_line=str2num(last_line_txt);
         last_file_time=last_line(1); % last register time
         tot_num_cols=length(last_line); % total number of columns in the file
         fseek(fid,fix(filelength*(start_time/last_file_time)),'bof');
         findline_backwards(fid);
         vars=read_file_part(fid,start_time,end_time);
         fclose(fid);
         if last_file_time < start_time
            disp('Error: The specified start_time is no included in the file')
            return
         end
      else
         disp(['Cannot open log file: ' LOG_FILE]);
      end

   otherwise
      disp('You must specify 1 or 3 arguments');
      disp('varlog var_type [start_time end_time]');
      disp('type: help varlog for more help');
      return
end    

num_input_vars=3*num_joints;
num_state_vars=3*num_joints;
num_torque_vars=num_joints;
num_output_vars=2*num_joints;
num_learning_vars=2*num_joints;
num_error_vars=num_joints;

time_col=1;
consum_col=time_col+1;
spikes_col=consum_col+1;
input_col=spikes_col+1;
state_col=input_col+num_input_vars;
torque_col=state_col+num_state_vars;
output_col=torque_col+num_torque_vars;
learning_col=output_col+num_output_vars;
error_col=learning_col+num_learning_vars;
tot_num_cols=error_col+num_error_vars;

reg_times=vars(:,time_col);
last_1exec_reg=find(reg_times > reg_times(1)+single_exec_dur,1);
if isempty(last_1exec_reg)
   last_1exec_reg=length(reg_times);
end

switch(v)
 case 'ti'
   subplot(2,1,1),stairs(reg_times,vars(:,consum_col)*1e3)
   ylabel('consumed time (ms)')
   subplot(2,1,2),stairs(reg_times,vars(:,spikes_col))
   ylabel('generated spikes')
   xlabel('time (s)')

 case 'in'
   title('Input vars')
   for ivar=1:num_input_vars,
      subplot(3,num_input_vars/3,ivar),plot(reg_times(1:last_1exec_reg),vars(1:last_1exec_reg,input_col+ivar-1))
      cur_ax=axis;
      axis([reg_times(1) reg_times(last_1exec_reg) cur_ax(3) cur_ax(4)]);
      ylabel(['in:' num2str(ivar)])
   end
   xlabel('time (s)')

 case 'st'
   title('State vars')
   for ivar=1:num_state_vars,
      subplot(3,num_state_vars/3,ivar),plot(reg_times,vars(:,input_col+ivar-1),'b',reg_times,vars(:,state_col+ivar-1),'r')
      ylabel(['st:' num2str(ivar)])
   end
   xlabel('time (s)')

 case 'to'
   title('Torque vars')
   for ivar=1:num_torque_vars,
      subplot(num_torque_vars,1,ivar),plot(reg_times(1:last_1exec_reg),vars(1:last_1exec_reg,torque_col+ivar-1))
      cur_ax=axis;
      axis([reg_times(1) reg_times(last_1exec_reg) cur_ax(3) cur_ax(4)]);
      ylabel(['to:' num2str(ivar)])
   end
   xlabel('time (s)')

 case 'ou'
   title('Output vars')
   for ivar=1:num_output_vars/2,
      subplot(num_output_vars/2,2,(ivar-1)*2+1),plot(reg_times,vars(:,output_col+(ivar-1)*2),'b',reg_times,vars(:,output_col+(ivar-1)*2+1),'r')
      ylabel(['out:' num2str(ivar)])
      subplot(num_output_vars/2,2,(ivar-1)*2+2),plot(reg_times,vars(:,output_col+(ivar-1)*2)-vars(:,output_col+(ivar-1)*2+1))
   end
   xlabel('time (s)')

 case 'tt'
   title('Total torque')
   for ivar=1:num_torque_vars,
      subplot(num_torque_vars,1,ivar),plot(reg_times,vars(:,torque_col+ivar-1)+vars(:,output_col+(ivar-1)*2)-vars(:,output_col+(ivar-1)*2+1))
      ylabel(['tt:' num2str(ivar)])
   end
   xlabel('time (s)')

 case 'le'
   title('Learning vars')
   for ivar=1:num_learning_vars,
      subplot(num_learning_vars,1,ivar),plot(reg_times,vars(:,learning_col+ivar-1))
      ylabel(['ler:' num2str(ivar)])
   end
   xlabel('time (s)')

 case 'er'
   title('Error vars')
   for ivar=1:num_error_vars,
      subplot(num_error_vars,1,ivar),plot(reg_times,vars(:,error_col+ivar-1))
      ylabel(['err:' num2str(ivar)])
   end
   xlabel('time (s)')

 case 'ma'
   mae_evol=[];
   cur_traj_start=reg_times(1);
   while cur_traj_start < reg_times(end)
      cur_traj_end=cur_traj_start + single_exec_dur-mod(cur_traj_start,single_exec_dur);
      traj_reg_times_i=find(reg_times>=cur_traj_start & reg_times<cur_traj_end);
      tra_mae=0;
      for ivar=1:num_state_vars/3,
         tra_mae=tra_mae+mae(vars(traj_reg_times_i,state_col+ivar-1)-vars(traj_reg_times_i,input_col+ivar-1));
      end
      mae_evol=[mae_evol [cur_traj_end;tra_mae ]];
      cur_traj_start=cur_traj_end;
   end
   plot(mae_evol(1,:),mae_evol(2,:));
   title('MAE evolution')
   xlabel('time (s)')
end

% READ_LINE_TIME gets the time of the next register from the simulation-log file.
%    REGTIME = READ_FILE_PART(FID) advances the file position indicator in the
%    file associated with the given FID to the beginning of the next text
%    line, then read and returns that register's time.
   function regtime=read_line_time(fid)
      time_read=0;
      regtime=-1;
      while time_read==0
         [regtime,time_read]=fscanf(fid,'%g',1);
         if time_read==0
            fgetl(fid);
         end
      end
   end

% FINDLINE_BACKWARDS move the a file pointer back to the beginning of the
% previous text line.
%    FINDLINE_BACKWARDS(FID) repositions the file position indicator in the
%    file associated with the given FID. FINDLINE_BACKWARDS sets the
%    position indicator to the byte at beginning of the line before the 
%    current one.
   function findline_backwards(fid)
      newline=sprintf('\n');
      tchar=' '; % look for the current-file-line start position
      while ~isempty(tchar) && isempty(strfind(tchar,newline))
         if fseek(fid,-2,'cof')==-1
             break
         end
         tchar = fscanf(fid,'%1c',1);
      end
   end

% READ_FILE_PART loads registers from the simulation-log file.
%    REGS = READ_FILE_PART(FID,STARTTIME,ENDTIME) returns the registers of
%    a file associated with file identifier FID as a MATLAB matrix. Only
%    the registers from STARTTIME to ENDTIME are loaded.
   function regs=read_file_part(fid,starttime,endtime)
      disp(' Searching for specified registers in the file...')
      while read_line_time(fid) > starttime
         findline_backwards(fid); % go back to the current line start
         fseek(fid,-1,'cof'); % jump before \n
         findline_backwards(fid); % go back to the previous line start
      end
      findline_backwards(fid);
      while read_line_time(fid) < starttime
         fgetl(fid); % jump after \n
      end
      findline_backwards(fid);      
      disp(' Loading registers from the file...')
      disp(' 00%')
      app_regs_size=(endtime-starttime)/sim_slot_time;  % estimated size for regs
      regs=zeros(ceil(app_regs_size),tot_num_cols); % allocate matrix memory (for execution time optmization)
      regs_size=0;
      cur_file_time=starttime;
      tline=' ';
      while cur_file_time < endtime && ischar(tline) && ~isempty(tline)
         tline=fgetl(fid);
         if ischar(tline) && ~isempty(tline) && isempty(~strfind(tline,'%'))
            vline=str2num(tline);
            regs_size=regs_size+1;
            regs(regs_size,:)=vline;
            cur_file_time=vline(1);
            if mod(regs_size,fix(app_regs_size/100)) == 0
               fprintf(1,'\b\b\b\b% 3.f%%',(regs_size/app_regs_size)*100);
            end
         end
      end
      regs=regs(1:regs_size,:); % trim array in case of initial overdimensioning
      fprintf(1,'\b\b\b\b100%%\n');
   end

end
