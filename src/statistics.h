#ifndef STATISTICS_H
#define STATISTICS_H

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>

namespace Statistics {

// Base abstract class
class AbstractStatsCounter {
 public:
  AbstractStatsCounter(const std::string &name,
                       const std::string &description = "")
      : m_name(name), m_description(description) {}
  virtual const std::string &name() const { return m_name; }
  virtual const std::string &description() const { return m_description; }
  virtual std::string csv_header_string() const = 0;
  virtual std::string csv_value_string() const = 0;
  // Convert counter name and value to a string
  virtual std::string to_string() const {
    return this->name() + " = " + csv_value_string();
  }

 protected:
  std::string m_name;
  std::string m_description;
};

// Generic stats counter with type parameter
template <typename RawCounterType>
class GenericStatsCounter : public AbstractStatsCounter {
 public:
  GenericStatsCounter(const std::string &name,
                      const std::string &description = "")
      : AbstractStatsCounter(name, description) {
    static_assert(std::is_arithmetic_v<RawCounterType>,
                  "RawCounterType must be an arithmetic type");
  }
  GenericStatsCounter(const GenericStatsCounter<RawCounterType> &other)
      : AbstractStatsCounter(other) {}
  ~GenericStatsCounter() {}

  // Get counter value
  virtual const RawCounterType &value() const = 0;
  // Reset counter value
  virtual void reset() = 0;

  // Overloading operators
  virtual GenericStatsCounter<RawCounterType> &operator=(
      const GenericStatsCounter<RawCounterType> &other) = 0;
  virtual GenericStatsCounter<RawCounterType> &operator++() = 0;
  virtual GenericStatsCounter<RawCounterType> &operator--() = 0;
  virtual GenericStatsCounter<RawCounterType> &operator+=(
      const GenericStatsCounter<RawCounterType> &other) = 0;
  virtual GenericStatsCounter<RawCounterType> &operator-=(
      const GenericStatsCounter<RawCounterType> &other) = 0;
  virtual GenericStatsCounter<RawCounterType> &operator*=(
      const GenericStatsCounter<RawCounterType> &other) = 0;
  virtual GenericStatsCounter<RawCounterType> &operator/=(
      const GenericStatsCounter<RawCounterType> &other) = 0;
  virtual GenericStatsCounter<RawCounterType> &operator=(
      RawCounterType value) = 0;
  virtual GenericStatsCounter<RawCounterType> &operator+=(
      RawCounterType value) = 0;
  virtual GenericStatsCounter<RawCounterType> &operator-=(
      RawCounterType value) = 0;
  virtual GenericStatsCounter<RawCounterType> &operator*=(
      RawCounterType value) = 0;
  virtual GenericStatsCounter<RawCounterType> &operator/=(
      RawCounterType value) = 0;

  // Convert counter name to a CSV header string
  virtual std::string csv_header_string() const override { return name(); }
  // Convert counter value to a string
  virtual std::string csv_value_string() const override {
    return std::to_string(value());
  }
};

template <typename RawCounterType>
class SingleStatsCounter : public GenericStatsCounter<RawCounterType> {
 public:
  SingleStatsCounter(const std::string &name,
                     const std::string &description = "",
                     RawCounterType initial_value = 0)
      : GenericStatsCounter<RawCounterType>(name, description),
        m_counter(initial_value),
        m_initial_value(initial_value) {}
  SingleStatsCounter(const SingleStatsCounter<RawCounterType> &other)
      : GenericStatsCounter<RawCounterType>(other), m_counter(other.value()) {}
  ~SingleStatsCounter() {}

  // Get counter value
  virtual const RawCounterType &value() const override { return m_counter; }
  // Reset counter value
  virtual void reset() override { m_counter = m_initial_value; }

  virtual SingleStatsCounter<RawCounterType> &operator=(
      const GenericStatsCounter<RawCounterType> &other) override {
    m_counter = other.value();
    return *this;
  }
  virtual SingleStatsCounter<RawCounterType> &operator++() override {
    m_counter++;
    return *this;
  }
  virtual SingleStatsCounter<RawCounterType> &operator--() override {
    m_counter--;
    return *this;
  }
  virtual SingleStatsCounter<RawCounterType> &operator+=(
      const GenericStatsCounter<RawCounterType> &other) override {
    m_counter += other.value();
    return *this;
  }
  virtual SingleStatsCounter<RawCounterType> &operator-=(
      const GenericStatsCounter<RawCounterType> &other) override {
    m_counter -= other.value();
    return *this;
  }
  virtual SingleStatsCounter<RawCounterType> &operator*=(
      const GenericStatsCounter<RawCounterType> &other) override {
    m_counter *= other.value();
    return *this;
  }
  virtual SingleStatsCounter<RawCounterType> &operator/=(
      const GenericStatsCounter<RawCounterType> &other) override {
    m_counter /= other.value();
    return *this;
  }
  virtual SingleStatsCounter<RawCounterType> &operator=(
      RawCounterType value) override {
    m_counter = value;
    return *this;
  }
  virtual SingleStatsCounter<RawCounterType> &operator+=(
      RawCounterType value) override {
    m_counter += value;
    return *this;
  }
  virtual SingleStatsCounter<RawCounterType> &operator-=(
      RawCounterType value) override {
    m_counter -= value;
    return *this;
  }
  virtual SingleStatsCounter<RawCounterType> &operator*=(
      RawCounterType value) override {
    m_counter *= value;
    return *this;
  }
  virtual SingleStatsCounter<RawCounterType> &operator/=(
      RawCounterType value) override {
    m_counter /= value;
    return *this;
  }

  // Roll-ups have following calculated quantities as built-in sub-metrics
  // FUTURE: These additional roll-ups could be useful
  // GenericStatsCounter peak_sustained() const;
  // GenericStatsCounter peak_sustained_active() const;
  // GenericStatsCounter peak_sustained_elapsed() const;
  // GenericStatsCounter per_second() const;
  // GenericStatsCounter per_cycle_active() const;
  // GenericStatsCounter per_cycle_elapsed() const;
  // GenericStatsCounter pct_of_peak_sustained_active() const;
  // GenericStatsCounter pct_of_peak_sustained_elapsed() const;

 protected:
  RawCounterType m_counter;
  RawCounterType m_initial_value;
};

// Multi-unit stats counter
template <typename RawCounterType>
class MultiUnitStatsCounter : public GenericStatsCounter<RawCounterType> {
 public:
  MultiUnitStatsCounter(const std::string &name,
                        const std::string &description = "",
                        size_t num_units = 0, RawCounterType initial_value = 0)
      : GenericStatsCounter<RawCounterType>(name, description),
        m_num_units(num_units),
        m_initial_value(initial_value) {
    // Allocate memory for counter values
    m_counters = new RawCounterType[m_num_units];
    std::fill(m_counters, m_counters + m_num_units, initial_value);
  }

  MultiUnitStatsCounter(const MultiUnitStatsCounter<RawCounterType> &other)
      : GenericStatsCounter<RawCounterType>(other),
        m_num_units(other.m_num_units),
        m_initial_value(other.m_initial_value) {
    m_counters = new RawCounterType[m_num_units];
    std::copy(other.m_counters, other.m_counters + m_num_units, m_counters);
  }

  ~MultiUnitStatsCounter() { delete[] m_counters; }

  // Multi-unit stats method matching NCU like stats counter
  // Sub-metrics/roll-ups across all unit instances
  // https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-entities
  SingleStatsCounter<RawCounterType> sum() const {
    RawCounterType sum = std::accumulate(m_counters, m_counters + m_num_units,
                                         static_cast<RawCounterType>(0));
    return SingleStatsCounter<RawCounterType>(
        this->m_name + ".sum", this->m_description + " sum of all units", sum);
  }

  SingleStatsCounter<double> avg() const {
    RawCounterType sum = std::accumulate(m_counters, m_counters + m_num_units,
                                         static_cast<RawCounterType>(0));
    return SingleStatsCounter<double>(
        this->m_name + ".avg", this->m_description + " average of all units",
        static_cast<double>(sum) / m_num_units);
  }

  SingleStatsCounter<RawCounterType> min() const {
    auto minIterator = std::min_element(m_counters, m_counters + m_num_units);
    if (minIterator == m_counters + m_num_units) {
      throw std::runtime_error("MultiUnitStatsCounter: no valid units for min");
    }
    RawCounterType min = *minIterator;
    return SingleStatsCounter<RawCounterType>(
        this->m_name + ".min", this->m_description + " minimum of all units",
        min);
  }

  SingleStatsCounter<RawCounterType> max() const {
    auto maxIterator = std::max_element(m_counters, m_counters + m_num_units);
    if (maxIterator == m_counters + m_num_units) {
      throw std::runtime_error("MultiUnitStatsCounter: no valid units for max");
    }
    RawCounterType max = *maxIterator;
    return SingleStatsCounter<RawCounterType>(
        this->m_name + ".max", this->m_description + " maximum of all units",
        max);
  }

  // For all units
  // Get counter value
  virtual const RawCounterType &value() const override {
    throw std::runtime_error("MultiUnitStatsCounter: not implemented");
  }
  // Reset counter value
  virtual void reset() override {
    std::fill(m_counters, m_counters + m_num_units, this->m_initial_value);
  }

  // For individual unit
  void check_valid_unit_index(size_t unit_index) const {
    if (unit_index >= m_num_units) {
      throw std::out_of_range("MultiUnitStatsCounter: unit index out of range");
    }
  }
  virtual const RawCounterType &value(size_t unit_index) const {
    check_valid_unit_index(unit_index);
    return m_counters[unit_index];
  }
  virtual void reset(size_t unit_index) {
    check_valid_unit_index(unit_index);
    m_counters[unit_index] = m_initial_value;
  }

  virtual MultiUnitStatsCounter<RawCounterType> &operator=(
      const GenericStatsCounter<RawCounterType> &other) override {
    std::copy(static_cast<const MultiUnitStatsCounter<RawCounterType> &>(other)
                  .m_counters,
              static_cast<const MultiUnitStatsCounter<RawCounterType> &>(other)
                      .m_counters +
                  m_num_units,
              m_counters);
    return *this;
  }
  virtual MultiUnitStatsCounter<RawCounterType> &operator++() override {
    std::for_each(m_counters, m_counters + m_num_units,
                  [](RawCounterType &counter) { counter++; });
    return *this;
  }
  virtual MultiUnitStatsCounter<RawCounterType> &operator--() override {
    std::for_each(m_counters, m_counters + m_num_units,
                  [](RawCounterType &counter) { counter--; });
    return *this;
  }
  virtual MultiUnitStatsCounter<RawCounterType> &operator+=(
      const GenericStatsCounter<RawCounterType> &other) override {
    std::transform(
        m_counters, m_counters + m_num_units,
        static_cast<const MultiUnitStatsCounter<RawCounterType> &>(other)
            .m_counters,
        m_counters, std::plus<RawCounterType>());
    return *this;
  }
  virtual MultiUnitStatsCounter<RawCounterType> &operator-=(
      const GenericStatsCounter<RawCounterType> &other) override {
    std::transform(
        m_counters, m_counters + m_num_units,
        static_cast<const MultiUnitStatsCounter<RawCounterType> &>(other)
            .m_counters,
        m_counters, std::minus<RawCounterType>());
    return *this;
  }
  virtual MultiUnitStatsCounter<RawCounterType> &operator*=(
      const GenericStatsCounter<RawCounterType> &other) override {
    std::transform(
        m_counters, m_counters + m_num_units,
        static_cast<const MultiUnitStatsCounter<RawCounterType> &>(other)
            .m_counters,
        m_counters, std::multiplies<RawCounterType>());
    return *this;
  }
  virtual MultiUnitStatsCounter<RawCounterType> &operator/=(
      const GenericStatsCounter<RawCounterType> &other) override {
    std::transform(
        m_counters, m_counters + m_num_units,
        static_cast<const MultiUnitStatsCounter<RawCounterType> &>(other)
            .m_counters,
        m_counters, std::divides<RawCounterType>());
    return *this;
  }
  virtual MultiUnitStatsCounter<RawCounterType> &operator=(
      RawCounterType value) override {
    std::fill(m_counters, m_counters + m_num_units, value);
    return *this;
  }
  virtual MultiUnitStatsCounter<RawCounterType> &operator+=(
      RawCounterType value) override {
    std::for_each(m_counters, m_counters + m_num_units,
                  [value](RawCounterType &counter) { counter += value; });
    return *this;
  }
  virtual MultiUnitStatsCounter<RawCounterType> &operator-=(
      RawCounterType value) override {
    std::for_each(m_counters, m_counters + m_num_units,
                  [value](RawCounterType &counter) { counter -= value; });
    return *this;
  }
  virtual MultiUnitStatsCounter<RawCounterType> &operator*=(
      RawCounterType value) override {
    std::for_each(m_counters, m_counters + m_num_units,
                  [value](RawCounterType &counter) { counter *= value; });
    return *this;
  }
  virtual MultiUnitStatsCounter<RawCounterType> &operator/=(
      RawCounterType value) override {
    std::for_each(m_counters, m_counters + m_num_units,
                  [value](RawCounterType &counter) { counter /= value; });
    return *this;
  }

  // Support indexing operator for individual unit, no bound check for
  // faster performance
  RawCounterType &operator[](size_t unit_index) {
    return m_counters[unit_index];
  }

  // Bound check indexing .at()
  RawCounterType &at(size_t unit_index) {
    check_valid_unit_index(unit_index);
    return m_counters[unit_index];
  }

  // Convert stats to a CSV header string
  virtual std::string csv_header_string() const override {
    std::string result = "";
    for (size_t i = 0; i < m_num_units; i++) {
      result += this->name() + "_" + std::to_string(i);
      if (i < m_num_units - 1) {
        result += ",";
      }
    }
    return result;
  }

  // Convert stats to a CSV value string
  virtual std::string csv_value_string() const override {
    std::string result = "";
    for (size_t i = 0; i < m_num_units; i++) {
      result += std::to_string(m_counters[i]);
      if (i < m_num_units - 1) {
        result += ",";
      }
    }
    return result;
  }

 protected:
  // Number of units in this counter, constant after construction
  size_t m_num_units;

  // Initial value for all units
  RawCounterType m_initial_value;

  // Pointer to the array of raw counters, dynamically allocated
  RawCounterType *m_counters;
};

// Common type counter definition
using UInt64SingleStatsCounter = SingleStatsCounter<uint64_t>;
using UInt64MultiUnitStatsCounter = MultiUnitStatsCounter<uint64_t>;
using FloatSingleStatsCounter = SingleStatsCounter<float>;
using FloatMultiUnitStatsCounter = MultiUnitStatsCounter<float>;

}  // namespace Statistics

#endif  // STATISTICS_H