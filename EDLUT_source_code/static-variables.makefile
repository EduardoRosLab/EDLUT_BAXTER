################################################################################
################# - MAKEFILE STATIC VARIABLES - ################################
################################################################################

exe-sources   := ${sources} ${exe-source-file}
rtexe-sources   := ${sources} ${rtexe-source-file}
step-sources   := ${sources} ${step-source-file}
precision-sources   := ${sources} ${prec-source-file}

objects       := $(filter %.o,$(subst   .c,.o,$(sources)))
objects       += $(filter %.o,$(subst  .cc,.o,$(sources)))
objects       += $(filter %.o,$(subst .cpp,.o,$(sources)))
objects       += $(filter %.o,$(subst .cu,.o,$(sources)))
dependencies  := $(subst .o,.d,$(objects))

exe-objects       := $(filter %.o,$(subst   .c,.o,$(exe-sources)))
exe-objects       += $(filter %.o,$(subst  .cc,.o,$(exe-sources)))
exe-objects       += $(filter %.o,$(subst .cpp,.o,$(exe-sources)))
exe-objects       += $(filter %.o,$(subst .cu,.o,$(exe-sources)))
exe-dependencies  := $(subst .o,.d,$(exe-objects))

rtexe-objects       := $(filter %.o,$(subst   .c,.o,$(rtexe-sources)))
rtexe-objects       += $(filter %.o,$(subst  .cc,.o,$(rtexe-sources)))
rtexe-objects       += $(filter %.o,$(subst .cpp,.o,$(rtexe-sources)))
rtexe-objects       += $(filter %.o,$(subst .cu,.o,$(rtexe-sources)))
rtexe-dependencies  := $(subst .o,.d,$(rtexe-objects))

step-objects       := $(filter %.o,$(subst   .c,.o,$(step-sources)))
step-objects       += $(filter %.o,$(subst  .cc,.o,$(step-sources)))
step-objects       += $(filter %.o,$(subst .cpp,.o,$(step-sources)))
step-objects       += $(filter %.o,$(subst .cu,.o,$(step-sources)))
step-dependencies  := $(subst .o,.d,$(step-objects))

precision-objects       := $(filter %.o,$(subst   .c,.o,$(precision-sources)))
precision-objects       += $(filter %.o,$(subst  .cc,.o,$(precision-sources)))
precision-objects       += $(filter %.o,$(subst .cpp,.o,$(precision-sources)))
precision-objects       += $(filter %.o,$(subst .cu,.o,$(precision-sources)))
precision-dependencies  := $(subst .o,.d,$(precision-objects))


libtarget     := $(libdir)/lib$(packagename).a
exetarget     := $(bindir)/$(packagename)
rtexetarget   := $(bindir)/RealTimeEDLUTKernel
steptarget     := $(bindir)/stepbystep
precisiontarget := $(bindir)/precisiontest
pkgconfigfile := $(packagename).pc


automakefile := make.auto
