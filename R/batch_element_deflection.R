#' Batch Element Deflection
#'
#' @param df a dataframe that must have three columns named: actor, behavior, and object
#' that correspond to the elements of your event. These must match with identities and behaviors
#' in the US 2015 dictionary.
#'
#' @return a dataframe in long format with a variable indicating the amount of deflection produced by each
#' element
#'
#' @importFrom dplyr mutate
#' @importFrom dplyr rowwise
#' @importFrom dplyr %>%
#' @importFrom tidyr unnest
#'
#' @export


batch_element_deflection <- function(df) {
                      df_res <- df %>%
                        rowwise() %>%
                        mutate(el_def = list(element_deflection(actor, behavior, object))) %>%
                        unnest(el_def)

                      return(df_res)
                  }
