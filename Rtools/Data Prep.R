library(data.table)
library(rio)

ilec.dat <- rio::import(file=srcPath,
                        format=",",
                        setclass="data.table")

ilec.dat[,Gender:=factor(Gender,levels=c('Male','Female'))]
ilec.dat[,Smoker_Status:=factor(Smoker_Status)]
ilec.dat[,Insurance_Plan:=factor(Insurance_Plan)]
ilec.dat[,Age_Basis:=factor(Age_Basis)]

ilec.dat[,Face_Amount_Band:=factor(Face_Amount_Band,
                                   levels=c("    1-9999",
                                            "   10000-24999",
                                            "   25000-49999",
                                            "   50000-99999",
                                            "  100000-249999",
                                            "  250000-499999",
                                            "  500000-999999",
                                            " 1000000-2499999",
                                            " 2500000-4999999",
                                            " 5000000-9999999",
                                            "10000000+"))]

ilec.dat[,Number_Of_Preferred_Classes:=as.character(Number_Of_Preferred_Classes)]
ilec.dat[is.na(Number_Of_Preferred_Classes),Number_Of_Preferred_Classes:="U"]
ilec.dat[,Number_Of_Preferred_Classes:=factor(Number_Of_Preferred_Classes,
                                              levels=c("U","2","3","4"))]

ilec.dat[,Preferred_Class:=as.character(Preferred_Class)]
ilec.dat[is.na(Preferred_Class),Preferred_Class:="U"]
ilec.dat[,Preferred_Class:=factor(Preferred_Class,
                                  levels=c("U","1","2","3","4"))]

ilec.dat[,SOA_Anticipated_Level_Term_Period:=factor(SOA_Anticipated_Level_Term_Period)]
ilec.dat[,SOA_Guaranteed_Level_Term_Period:=factor(SOA_Guaranteed_Level_Term_Period)]
ilec.dat[,SOA_Post_level_Term_Indicator:=factor(SOA_Post_level_Term_Indicator)]
ilec.dat[,Select_Ultimate_Indicator:=factor(Select_Ultimate_Indicator)]

# Groupings

# Issue Age
# Appendix A, C
# Appendix JA
ilec.dat[, IA_Grp_1:=cut(Issue_Age,
                         breaks=c(-1,0,4,9,17,24,
                                  29,34,39,49,59,
                                  69,79,1000),
                         labels=c('0','1-4',
                                  '5-9',
                                  '10-17',
                                  '18-24',
                                  '25-29',
                                  '30-34',
                                  '35-39',
                                  '40-49',
                                  '50-59',
                                  '60-69',
                                  '70-79',
                                  '80+'))]
# Appendix E1-E8
ilec.dat[, IA_Grp_2:=cut(Issue_Age,
                         breaks=c(-1,17,
                                  29,39,49,59,
                                  69,1000),
                         labels=c('0-17',
                                  '18-29',
                                  '30-39',
                                  '40-49',
                                  '50-59',
                                  '60-69',
                                  '70+'))]
# Appendix F1-F4
ilec.dat[, IA_Grp_3:=cut(Issue_Age,
                         breaks=c(-1,17,seq(24,94,5),1000),
                         labels=c(paste(c(0,18,seq(29,94,5)-4),
                                      c(17,24,seq(29,94,5)),sep="-"),'95+'))]
# Appendix L1, L2
ilec.dat[,IA_Grp_4:=cut(Issue_Age,
                        breaks=c(-1,17,39,59,1000),
                        labels=c('0-17','18-39',
                                 '40-59','60+'))]
# Duration
# Appendix A, C
# Appendix F1-F4
# Appendix H
# Appendix JA
ilec.dat[,Dur_Grp_1 := cut(Duration,
                           breaks=c(-1,1,2,3,5,10,
                                    15,20,25,1000),
                           labels=c('1','2','3',
                                    '4-5','6-10',
                                    '11-15','16-20',
                                    '21-25','26+'))]
# Appendix L1, L2
ilec.dat[,Dur_Grp_2:=cut(Duration,
                         breaks=c(-1,25,1000),
                         labels=c('1-25','26+'))]
# Appendix L1, L2
ilec.dat[,Dur_Grp_3:=cut(Duration,
                         breaks=c(-1,5,10,
                                  15,20,25,1000),
                         labels=c('1-5','6-10',
                                  '11-15','16-20',
                                  '21-25','26+'))]
# Attained Age
# Appendix B
ilec.dat[,AA_Grp_1:=cut(Attained_Age,
                        breaks=c(-1,17,24,29,
                                 34,39,49,59,
                                 69,79,89,1000),
                        labels=c('0-17','18-24',
                                 '25-29',
                                 '30-34',
                                 '35-39',
                                 '40-49',
                                 '50-59',
                                 '60-69',
                                 '70-79',
                                 '80-89',
                                 '90+'))]
# Appendix OA1, OA2
ilec.dat[,AA_Grp_2:=cut(Attained_Age,
                        breaks=c(-1,64,69,79,89,1000),
                        labels=c('0-64','65-69',
                                 '70-79','80-89','90+'))]
# Face amount band
# Appendix A
ilec.dat[,FA_Band_1:=factor(Face_Amount_Band,
                                   levels=c("    1-9999",
                                            "   10000-24999",
                                            "   25000-49999",
                                            "   50000-99999",
                                            "  100000-249999",
                                            "  250000-499999",
                                            "  500000-999999",
                                            " 1000000-2499999",
                                            " 2500000-4999999",
                                            " 5000000-9999999",
                                            "10000000+"),
                                   labels=c("1-9,999",
                                            "10,000-24,999",
                                            "25,000-49,999",
                                            "50,000-99,999",
                                            "100,000-249,999",
                                            "250,000-499,999",
                                            "500,000-999,999",
                                            "1,000,000-2,499,999",
                                            "2,500,000-4,999,999",
                                            "5,000,000-9,999,999",
                                            "10,000,000+"))]
# Appendix B
ilec.dat[,FA_Band_2 := as.character(Face_Amount_Band)]

ilec.dat[!FA_Band_2 %in% c("    1-9999",
                                 "   10000-24999",
                                 "   25000-49999",
                                 "   50000-99999"),
         FA_Band_2 := '  100000+']

ilec.dat[,FA_Band_2:=factor(FA_Band_2,
                            levels=c("    1-9999",
                                     "   10000-24999",
                                     "   25000-49999",
                                     "   50000-99999",
                                     "  100000+"),
                            labels=c("1-9,999",
                                     "10,000-24,999",
                                     "25,000-49,999",
                                     "50,000-99,999",
                                     "100,000+"))]

# Appendix I
# Appendix K1,K2
ilec.dat[,FA_Band_3 := as.character(Face_Amount_Band)]
ilec.dat[FA_Band_3 %in% c("    1-9999",
                          "   10000-24999",
                          "   25000-49999",
                          "   50000-99999"),
         FA_Band_3:='    1-99999']
ilec.dat[FA_Band_3 %in% c(" 2500000-4999999",
                          " 5000000-9999999",
                          "10000000+"),
         FA_Band_3 := ' 2500000+']
ilec.dat[,FA_Band_3:=factor(FA_Band_3,
                           levels=c('    1-99999',
                                    "  100000-249999",
                                    "  250000-499999",
                                    "  500000-999999",
                                    " 1000000-2499999",
                                    ' 2500000+'),
                           labels=c('1-99,999',
                                    "100,000-249,999",
                                    "250,000-499,999",
                                    "500,000-999,999",
                                    "1,000,000-2,499,999",
                                    '2,500,000+'))]

# Appendix JA
ilec.dat[,FA_Band_4:=as.character(Face_Amount_Band)]
ilec.dat[FA_Band_4 %in% c(" 1000000-2499999",
                           " 2500000-4999999",
                           " 5000000-9999999",
                           "10000000+"),
         FA_Band_4:=' 1000000+']
ilec.dat[,FA_Band_4:=factor(FA_Band_4,
                            levels=c("    1-9999",
                                     "   10000-24999",
                                     "   25000-49999",
                                     "   50000-99999",
                                     "  100000-249999",
                                     "  250000-499999",
                                     "  500000-999999",
                                     " 1000000+"),
                            labels=c("1-9,999",
                                     "10,000-24,999",
                                     "25,000-49,999",
                                     "50,000-99,999",
                                     "100,000-249,999",
                                     "250,000-499,999",
                                     "500,000-999,999",
                                     "1,000,000+"))]

# Issue Year
# Appendix H
ilec.dat[,IY_Grp_1:=cut(Issue_Year,
                        breaks=c(1900,1989,1999,2009,2016),
                        labels=c('1900-1989',
                                 '1990-1999',
                                 '2000-2009',
                                 '2010-2016'))]

# Insurance Plan
# Appendix G
ilec.dat[,Insurance_Plan_2:=as.character(Insurance_Plan)]
ilec.dat[Insurance_Plan_2 %in% c(" ULSG"," VLSG"),Insurance_Plan_2 := "xLSG"]
ilec.dat[,Insurance_Plan_2 := factor(Insurance_Plan_2,
                                     levels=c(" Term"," Perm"," UL"," VL","xLSG","Other"),
                                     labels=c("Term Insurance Plans","Traditional Whole Life Plans","Universal Life Plans","Variable Life Plans","ULSG and VLSG Plans","Other"))]

ilec.dat[,V1:=NULL]

#saveRDS(object=ilec.dat,
#        file=localRDSPath)


write_fst(x=ilec.dat,
          path=localRDSPath)
