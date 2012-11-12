#ifndef TOTALLY_CORRECTIVE_BOOSTING_BOOSTERS_FACTORY_HPP
#define TOTALLY_CORRECTIVE_BOOSTING_BOOSTERS_FACTORY_HPP

#include "ConfigFile.hpp"

#include <boost/shared_ptr.hpp>

#include <vector>
#include <iosfwd>

namespace totally_corrective_boosting
{

// forward declarations
class AbstractBooster;
class AbstractOracle;

AbstractBooster *new_booster_instance(const ConfigFile &config,
                                      const std::vector<int> &labels,
                                      const boost::shared_ptr<AbstractOracle> &oracle,
                                      std::ostream &log_stream = std::cout);

} // end of namespace totally_corrective_boosting

#endif // TOTALLY_CORRECTIVE_BOOSTING_BOOSTERS_FACTORY_HPP
