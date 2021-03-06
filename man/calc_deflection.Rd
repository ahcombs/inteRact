% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/calc_deflection.R
\name{calc_deflection}
\alias{calc_deflection}
\title{Compute Deflection for an Event}
\usage{
calc_deflection(act, beh, obj, dictionary = "us")
}
\arguments{
\item{act}{lowercase string corresponding to the actor identity}

\item{beh}{lowercase string corresponding to the behavior term}

\item{obj}{lowercase string corresponding to the object identity}

\item{dictionary}{which dictionary to use, currently set to "us"}
}
\value{
calculated deflection for the event (single number)
}
\description{
This function calculates the deflection for an Actor, Behavior, Object event.
It assumes that the first input is an identity corresponding to the actor,
the second, the behavior, and last, the object. Each of these terms must be
in the US 2015 dictionary.
}
\examples{

calc_deflection("ceo", "advise", "benefactor")
}
