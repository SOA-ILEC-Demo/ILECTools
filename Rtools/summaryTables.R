# Delivers cuts of the data as prescribed

summaryTable <- function(.data,
                         .dimList=NULL) {
  
  if(!is.data.table(.data) | !is.data.frame(.data))
    stop("Function summaryTable requires a data.table or data.frame for the .data argument.")
  
  if(!is.data.table(.data))
    .data <- data.table(.data)
  
  if(is.null(.dimList)) {
    retVal <- .data[Observation_Year <= 2017,
                       .(Number_Of_Deaths=sum(Number_Of_Deaths),
                         Expected_Death_QX2015VBT_by_Policy=sum(Expected_Death_QX2015VBT_by_Policy),
                         A_E_Policy=sum(Number_Of_Deaths)/sum(Expected_Death_QX2015VBT_by_Policy),
                         Policies_Exposed=sum(Policies_Exposed),
                         Policies_Exposed_Prop=sum(Policies_Exposed),
                         Death_Claim_Amount=sum(Death_Claim_Amount),
                         Expected_Death_QX2015VBT_by_Amount=sum(Expected_Death_QX2015VBT_by_Amount),
                         A_E_Amount=sum(Death_Claim_Amount)/sum(Expected_Death_QX2015VBT_by_Amount),
                         Amount_Exposed=sum(Amount_Exposed),
                         Amount_Exposed_Prop=sum(Amount_Exposed)),
                       by=eval(.dimList[[1]])
    ][,
      `:=`(Policies_Exposed_Prop=Policies_Exposed/sum(Policies_Exposed),
           Amount_Exposed_Prop=Amount_Exposed/sum(Amount_Exposed)
      )] %>% 
      gt() %>%
      tab_header(
        title="Summary claim statistics vs. 2015VBT"
      )
  }
  else {
    retVal <- .data[Observation_Year <= 2017,
                     .(Number_Of_Deaths=sum(Number_Of_Deaths),
                       Expected_Death_QX2015VBT_by_Policy=sum(Expected_Death_QX2015VBT_by_Policy),
                       A_E_Policy=sum(Number_Of_Deaths)/sum(Expected_Death_QX2015VBT_by_Policy),
                       Policies_Exposed=sum(Policies_Exposed),
                       Policies_Exposed_Prop=sum(Policies_Exposed),
                       Death_Claim_Amount=sum(Death_Claim_Amount),
                       Expected_Death_QX2015VBT_by_Amount=sum(Expected_Death_QX2015VBT_by_Amount),
                       A_E_Amount=sum(Death_Claim_Amount)/sum(Expected_Death_QX2015VBT_by_Amount),
                       Amount_Exposed=sum(Amount_Exposed),
                       Amount_Exposed_Prop=sum(Amount_Exposed)),
                     by=eval(.dimList[[1]])
  ][,
    `:=`(Policies_Exposed_Prop=Policies_Exposed/sum(Policies_Exposed),
         Amount_Exposed_Prop=Amount_Exposed/sum(Amount_Exposed)
    )] %>% 
    gt() %>%
    tab_header(
      title="Summary claim statistics vs. 2015VBT",
      subtitle=paste0("by ",paste(.dimList[[2]],collapse=", "))
    )
  } 
  
  retVal <- retVal %>%
    tab_spanner(
      label="by count",
      columns = c(
        Number_Of_Deaths,
        Expected_Death_QX2015VBT_by_Policy,
        A_E_Policy,
        Policies_Exposed,
        Policies_Exposed_Prop
      )
    ) %>%
    tab_spanner(
      label="by amount",
      columns = c(
        Death_Claim_Amount,
        Expected_Death_QX2015VBT_by_Amount,
        A_E_Amount,
        Amount_Exposed,
        Amount_Exposed_Prop
      )
    ) %>%
    fmt_integer(
      columns = Number_Of_Deaths
    ) %>%
    fmt_percent(
      columns=c(A_E_Policy,Policies_Exposed_Prop,
                A_E_Amount,Amount_Exposed_Prop),
      decimals = 1
    ) %>%
    fmt_number(
      columns=c(Expected_Death_QX2015VBT_by_Policy)
    ) %>% 
    fmt_number(
      columns=c(Policies_Exposed),
      suffixing = T
    ) %>% 
    fmt_currency(
      columns=c(
        Death_Claim_Amount,
        Expected_Death_QX2015VBT_by_Amount,
        Amount_Exposed),
      suffixing = T
    ) %>% 
    cols_label(
      Number_Of_Deaths="Actual Deaths",
      Expected_Death_QX2015VBT_by_Policy="Tabular Deaths",
      A_E_Policy="Actual-to-Tabular",
      Policies_Exposed="Policies Exposed",
      Policies_Exposed_Prop="Proportion Exposure",
      Death_Claim_Amount="Actual Amount",
      Expected_Death_QX2015VBT_by_Amount="Tabular Amount",
      A_E_Amount="Actual-to-Tabular",
      Amount_Exposed_Prop="Proportion Exposure"
    ) 
  
  if(!is.null(.dimList)) {
    retVal <- retVal %>% 
      cols_label(
        .list = purrr::set_names(as.list(.dimList[[2]]),
                                 .dimList[[1]])
      )
  }
  
  retVal
}
