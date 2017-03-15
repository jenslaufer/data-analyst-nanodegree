states_info <- read.csv('stateData.csv')

murder_states <- subset(states_info, murder > 10)

high_income <- states_info[states_info$income > 5000,]
