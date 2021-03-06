% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/closest_term.R
\name{closest_term}
\alias{closest_term}
\title{Closest Term}
\usage{
closest_term(e, p, a, dict = "us", term_typ, max_dist = 1)
}
\arguments{
\item{e}{a numeric variable equaling the Evaluation of the EPA location}

\item{p}{a numeric variable equaling the Potency of the EPA location}

\item{a}{a numeric variable equaling the Activity of the EPA location}

\item{dict}{a string indicating what dictionary you are using, currently set to 'us'}

\item{term_typ}{a string indicating whether you are looking for an identity, behavior, or modifier}

\item{max_dist}{a numeric variable equaling the maximum distance a term can be away from your EPA location}
}
\value{
a dataframe listing terms matching your search. Includes variables indicating element-wise distance and sum of squared differences.
}
\description{
A function to find the closest measured identity, behavior, or modifier to a specified E, P, A location.
}
\examples{
closest_term(1, 2.5, 3, term_typ = "identity")
}
