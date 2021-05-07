// Generated by gencpp from file mrobot_navigation/SbsResponse.msg
// DO NOT EDIT!


#ifndef MROBOT_NAVIGATION_MESSAGE_SBSRESPONSE_H
#define MROBOT_NAVIGATION_MESSAGE_SBSRESPONSE_H


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
struct SbsResponse_
{
  typedef SbsResponse_<ContainerAllocator> Type;

  SbsResponse_()
    : result(0)  {
    }
  SbsResponse_(const ContainerAllocator& _alloc)
    : result(0)  {
  (void)_alloc;
    }



   typedef uint8_t _result_type;
  _result_type result;





  typedef boost::shared_ptr< ::mrobot_navigation::SbsResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::mrobot_navigation::SbsResponse_<ContainerAllocator> const> ConstPtr;

}; // struct SbsResponse_

typedef ::mrobot_navigation::SbsResponse_<std::allocator<void> > SbsResponse;

typedef boost::shared_ptr< ::mrobot_navigation::SbsResponse > SbsResponsePtr;
typedef boost::shared_ptr< ::mrobot_navigation::SbsResponse const> SbsResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::mrobot_navigation::SbsResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::mrobot_navigation::SbsResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::mrobot_navigation::SbsResponse_<ContainerAllocator1> & lhs, const ::mrobot_navigation::SbsResponse_<ContainerAllocator2> & rhs)
{
  return lhs.result == rhs.result;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::mrobot_navigation::SbsResponse_<ContainerAllocator1> & lhs, const ::mrobot_navigation::SbsResponse_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace mrobot_navigation

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::mrobot_navigation::SbsResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::mrobot_navigation::SbsResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::mrobot_navigation::SbsResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::mrobot_navigation::SbsResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::mrobot_navigation::SbsResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::mrobot_navigation::SbsResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::mrobot_navigation::SbsResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "25458147911545c320c4c0a299eff763";
  }

  static const char* value(const ::mrobot_navigation::SbsResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x25458147911545c3ULL;
  static const uint64_t static_value2 = 0x20c4c0a299eff763ULL;
};

template<class ContainerAllocator>
struct DataType< ::mrobot_navigation::SbsResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "mrobot_navigation/SbsResponse";
  }

  static const char* value(const ::mrobot_navigation::SbsResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::mrobot_navigation::SbsResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uint8 result\n"
;
  }

  static const char* value(const ::mrobot_navigation::SbsResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::mrobot_navigation::SbsResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.result);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct SbsResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::mrobot_navigation::SbsResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::mrobot_navigation::SbsResponse_<ContainerAllocator>& v)
  {
    s << indent << "result: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.result);
  }
};

} // namespace message_operations
} // namespace ros

#endif // MROBOT_NAVIGATION_MESSAGE_SBSRESPONSE_H