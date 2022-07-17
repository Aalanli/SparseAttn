/*
template meta-programming classes for convenience in kernel specialization
*/

#pragma once
#include <boost/mpl/list.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/begin.hpp>
#include <boost/mpl/list_c.hpp>
#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/joint_view.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/integral_c.hpp>
#include <boost/mpl/size.hpp>

#include <string>
#include <unordered_map>
#include <map>
#include <iostream>


namespace fn_builder {

namespace mpl = boost::mpl;

namespace aux {

    template <typename Seq, template <class ...> class C, class... ArgsSoFar>
    struct ApplyArgs {
        using front_type = typename mpl::front<Seq>::type;
        using back_list  = typename mpl::pop_front<Seq>::type;
        using type = typename ApplyArgs<back_list, C, ArgsSoFar ..., front_type>::type;
    };

    template <template <class ...> class C, class... ArgsSoFar>
    struct ApplyArgs<mpl::l_end, C, ArgsSoFar ...> {
        using type = C<ArgsSoFar ...>;
    };

    template <typename Seq, typename Map, int N, template <class ...> class Fn>
    struct BuildFn_ {
        static void build_func_(Map &map) {
            using front = typename mpl::front<Seq>::type;
            using fn_t  = typename ApplyArgs<front, Fn>::type;
            map[fn_t::get_id()] = &fn_t::fn;
            using back = typename mpl::pop_front<Seq>::type;
            BuildFn_<back, Map, N+1, Fn>::build_func_(map);
        }
    };

    template <typename Map, int N, template <class ...> class Fn>
    struct BuildFn_<mpl::l_end, Map, N, Fn> {
        static void build_func_(Map &map) {}
    };
}


template <typename Seq, template <class ...> class Fn>
struct FnBuilder {
    const static int size = mpl::size<Seq>::value;
    using front = typename mpl::front<Seq>::type;
    using fn_t  = typename aux::ApplyArgs<front, Fn>::type;
    using fn_ptr = decltype(&fn_t::fn);

    using map_t = std::map<size_t, fn_ptr>;

    static map_t build_fn() {
        map_t map;
        aux::BuildFn_<Seq, map_t, 0, Fn>::build_func_(map);
        return map;
    }

    static void build_fn(map_t &map) {
        aux::BuildFn_<Seq, map_t, 0, Fn>::build_func_(map);
    }
};

}