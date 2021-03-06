#' Function to Generate the Re-identification of the Actor
#'
#' @param act lowercase string corresponding to the actor identity
#' @param beh lowercase string corresponding to the behavior term
#' @param obj lowercase string corresponding to the object identity
#' @param dictionary which dictionary to use, currently set to "us"
#'
#' @return dataframe with 3 columns corresponding to the EPA of optimal actor identity relabel
#' @importFrom tibble tibble
#' @export
#'
#' @examples
reidentify_actor <- function(act, beh, obj, dictionary = "us") {
          #calculate the transient impression of the event
          trans_imp_df <- transient_impression(act, beh, obj, dictionary = "us")

          #extract terms that are not A
          i_a <- extract_terms(elem = "A", trans_imp_df)

          #create actor selection matrix
          a_s <- create_select_mat("actor")

          #now which terms do not have actor in them
          i_s <- matrix(data = rep(1, 29), nrow = 29)
          i_3 <- as.matrix(c(1, 1, 1))
          g <- i_s - a_s %*% i_3
          g <- as.vector(g)

          #construct h matrix
          h <- construct_h_matrix()

          #term 1 of equation
          term1 <- t(a_s) %*% i_a %*% h %*% i_a %*% a_s
          term1 <- solve(term1)
          term1 <- -1*term1

          #term 2 of the equation
          term2 <- t(a_s) %*% i_a %*% h %*% i_a %*% g

          #final solution
          sol <- term1 %*% term2

          #put into nicer format
          actor_label <- tibble(E = sol[1],
                                 P = sol[2],
                                 A = sol[3])

          return(actor_label)
}
