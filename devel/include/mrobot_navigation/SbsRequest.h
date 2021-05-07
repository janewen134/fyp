// Generated by gencpp from file mrobot_navigation/SbsRequest.msg
// DO NOT EDIT!


#ifndef MROBOT_NAVIGATION_MESSAGE_SBSREQUEST_H
#define MROBOT_NAVIGATION_MESSAGE_SBSREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace mrobot_navigation
{
template <class ContainerAllocator>
struct SbsRequest_
{
  typedef SbsRequest_<ContainerAllocator> Type;

  SbsRequest_()
    {
    }
  SbsRequest_(const ContainerAllocator& _alloc)
    {
  (void)_alloc;
    }







  typedef boost::shared_ptr< ::mrobot_navigation::SbsRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::mrobot_navigation::SbsRequest_<ContainerAllocator> const> ConstPtr;

}; // struct SbsRequest_

typedef ::mrobot_navigation::SbsRequest_<std::allocator<void> > SbsRequest;

typedef boost::shared_ptr< ::mrobot_navigation::SbsRequest > SbsRequestPtr;
typedef boost::shared_ptr< ::mrobot_navigation::SbsRequest const> SbsRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::mrobot_navigation::SbsRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::mrobot_navigation::SbsRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


} // namespace mrobot_navigation

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::mrobot_navigation::SbsRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::mrobot_navigation::SbsRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::mrobot_navigation::SbsRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::mrobot_navigation::SbsRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::mrobot_navigation::SbsRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::mrobot_navigation::SbsRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::mrobot_navigation::SbsRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d41d8cd98f00b204e9800998ecf8427e";
  }

  static const char* value(const ::mrobot_navigation::SbsRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd41d8cd98f00b204ULL;
  static const uint64_t static_value2 = 0xe9800998ecf8427eULL;
};

template<class ContainerAllocator>
struct DataType< ::mrobot_navigation::SbsRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "mrobot_navigation/SbsRequest";
  }

  static const char* value(const ::mrobot_navigation::SbsRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::mrobot_navigation::SbsRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n"
;
  }

  static const char* value(const ::mrobot_navigation::SbsRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::mrobot_navigation::SbsRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream&, T)
    {}

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct SbsRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::mrobot_navigation::SbsRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream&, const std::string&, const ::mrobot_navigation::SbsRequest_<ContainerAllocator>&)
  {}
};

} // namespace message_operations
} // namespace ros

#endif // MROBOT_NAVIGATION_MESSAGE_SBSREQUEST_H
